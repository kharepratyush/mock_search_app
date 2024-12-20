import os
import time
import logging
from typing import Dict, Any
from datetime import datetime

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CosineSimilarityLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.training_args import BatchSamplers

from data_loader import load_data

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def configure_training_args(
    output_dir: str,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    fp16: bool = False,
    bf16: bool = False,
    eval_strategy: str = "no",
    eval_steps: int = None,
    save_strategy: str = "steps",
    save_steps: int = 100,
    save_total_limit: int = 2,
    logging_steps: int = 100,
    batch_sampler:str = 'no_duplicates',
    run_name: str = None
) -> SentenceTransformerTrainingArguments:
    """
    Configure training arguments for SentenceTransformerTrainer.

    Parameters
    ----------
    output_dir : str
        Directory to store the trained models and checkpoints.
    num_train_epochs : int, optional
        Number of training epochs.
    per_device_train_batch_size : int, optional
        Training batch size per device.
    per_device_eval_batch_size : int, optional
        Evaluation batch size per device.
    learning_rate : float, optional
        Learning rate for the optimizer.
    warmup_ratio : float, optional
        The ratio of total steps used for a linear warmup of LR.
    fp16 : bool, optional
        Whether to use FP16 training.
    bf16 : bool, optional
        Whether to use BF16 training.
    eval_strategy : str, optional
        Evaluation strategy ("no", "steps", or "epoch").
    eval_steps : int, optional
        Number of steps after which to run evaluation.
    save_strategy : str, optional
        Strategy to save checkpoints ("no", "steps", or "epoch").
    save_steps : int, optional
        Number of steps after which a checkpoint is saved.
    save_total_limit : int, optional
        Maximum number of checkpoints to keep.
    logging_steps : int, optional
        Number of steps after which logs are printed.
    batch_sampler : BatchSamplers, optional
        How to sample batches for training.
    run_name : str, optional
        A name for the experiment run (e.g., for W&B logging).

    Returns
    -------
    SentenceTransformerTrainingArguments
        Configured training arguments.
    """
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        batch_sampler=batch_sampler,
        run_name=run_name,
    )


def evaluate_model(model: SentenceTransformer, eval_dataset, evaluator_name: str) -> Dict[str, Any]:
    """
    Evaluate the model on a given dataset with a BinaryClassificationEvaluator.

    Parameters
    ----------
    model : SentenceTransformer
        The model to evaluate.
    eval_dataset : datasets.Dataset
        Evaluation dataset containing 'query', 'dish', and 'label' columns.
    evaluator_name : str
        A name for this evaluation run.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics.
    """
    logger.info(f"Evaluating model: {evaluator_name}")
    evaluator = BinaryClassificationEvaluator(
        sentences1=eval_dataset["query"],
        sentences2=eval_dataset["dish"],
        labels=eval_dataset["label"],
        name=evaluator_name,
    )
    results = evaluator(model)
    logger.info(f"Evaluation {evaluator_name} results: {results}")
    return results


def train_unlabeled_model(
    model_name: str,
    unlabeled_file: str,
    output_dir: str
):
    """
    Train a model on unlabeled data using MultipleNegativesRankingLoss.

    Parameters
    ----------
    model_name : str
        Pre-trained model name or path.
    unlabeled_file : str
        Path to unlabeled CSV dataset file.
    output_dir : str
        Directory to save trained model and artifacts.
    """
    logger.info("Loading unlabeled dataset...")
    unlabeled_data = load_data(unlabeled_file, dataset_type="unlabeled")
    logger.info(f"Unlabeled dataset loaded: {unlabeled_data}")

    logger.info("Initializing model...")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded: {model_name}")

    loss = MultipleNegativesRankingLoss(model)
    training_args = configure_training_args(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1000,
        per_device_eval_batch_size=1000,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # For MultipleNegativesRankingLoss
        run_name="unlabeled-training-run"
    )

    logger.info("Starting unlabeled training...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=unlabeled_data,
        loss=loss,
    )

    trainer.train()
    logger.info("Unlabeled training complete.")

    return model


def train_labeled_model(
    model: SentenceTransformer,
    labeled_file: str,
    eval_file: str,
    output_dir: str
):
    """
    Fine-tune the model on labeled data using CosineSimilarityLoss and evaluate.

    Parameters
    ----------
    model : SentenceTransformer
        The model to be further fine-tuned.
    labeled_file : str
        Path to labeled CSV dataset file.
    eval_file : str
        Path to evaluation CSV dataset file.
    output_dir : str
        Directory to save the fine-tuned model and artifacts.
    """
    logger.info("Loading labeled training dataset...")
    train_dataset = load_data(labeled_file, dataset_type="labeled")
    logger.info("Loading evaluation dataset...")
    eval_dataset_dict = load_data(eval_file, dataset_type="labeled", name="eval")
    eval_dataset = eval_dataset_dict["eval"]['train']
    logger.info(f"Evaluation dataset loaded: {eval_dataset}")

    loss = CosineSimilarityLoss(model)
    training_args = configure_training_args(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1000,
        per_device_eval_batch_size=1000,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        #batch_sampler='no_duplicates',
        run_name="labeled-training-run"
    )

    # Initial evaluation before fine-tuning
    evaluate_model(model, eval_dataset, "Initial Labeled Model on Labeled Data")

    logger.info("Starting labeled training...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],  # Assuming 'train' split is present
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=BinaryClassificationEvaluator(
            sentences1=eval_dataset["query"],
            sentences2=eval_dataset["dish"],
            labels=eval_dataset["label"],
            name="Labeled Model on Labeled Data Eval"
        ),
    )

    trainer.train()
    logger.info("Labeled training complete.")
    return model


if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    UNLABELED_FILE = "unlabeled_dataset.csv"
    LABELED_FILE = "labeled_dataset.csv"
    EVAL_FILE = "eval_labeled_dataset.csv"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    UNLABELED_OUTPUT_DIR = f"models/unlabeled_bge-small-en-v1.5_{TIMESTAMP}"
    LABELED_OUTPUT_DIR = f"models/labeled_bge-small-en-v1.5_{TIMESTAMP}"
    FINAL_OUTPUT_DIR = "models/final"

    # Ensure output directories exist
    os.makedirs(UNLABELED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LABELED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

    # Train on unlabeled data
    model = train_unlabeled_model(MODEL_NAME, UNLABELED_FILE, UNLABELED_OUTPUT_DIR)

    # Evaluate on labeled data after unlabeled training
    eval_dataset_dict = load_data(EVAL_FILE, dataset_type="labeled", name="eval")
    eval_dataset = eval_dataset_dict["eval"]['train']
    evaluate_model(model, eval_dataset, "Unlabeled Model on Labeled Data")

    # Train on labeled data
    model = train_labeled_model(model, LABELED_FILE, EVAL_FILE, LABELED_OUTPUT_DIR)

    # Save final model
    model.save_pretrained(FINAL_OUTPUT_DIR)
    logger.info(f"Final model saved to {FINAL_OUTPUT_DIR}")

    # Sleep for a moment to ensure logs are flushed
    time.sleep(1)
