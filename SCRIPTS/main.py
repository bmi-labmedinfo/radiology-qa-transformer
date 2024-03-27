from datasets import disable_caching
from transformers.utils import logging
from batch_truncation_models import *
from ex_post_combination_models import *
from fine_tuning import *

logging.set_verbosity(logging.WARNING)
disable_caching()

# to log in to wandb
#import wandb
#wandb.login(key="...")

from configuration import Config
config = Config()

# --------------------  BATCH TRUNCATION  --------------------------
# --- PRE-TRAINED MODEL
for k in np.arange(5):
    #k-fold for 5-CV
    bt_pt_model = BatchTruncation_PretrainedModel(k)
    bt_pt_model.validation()

# --- FINE-TUNED MODEL
for k in np.arange(5):
    #k-fold for 5-CV
    batch_truncation_finetuning(k)
    bt_ft_model = BatchTruncation_FinetunedModel(k_fold=k)
    bt_ft_model.validation()

# -------------------  EX-POST COMBINATION  ------------------------
# --- PRE-TRAINED MODEL
for k in np.arange(5):
    #k-fold for 5-CV
    epc_pt_model = ExPostCombination_PretrainedModel(k_fold=k)
    epc_pt_model.first_validation()
    epc_pt_model.reassembly_batches()

# --- EPC FINE-TUNED MODEL
for i in np.arange(5):
    ex_post_combination_finetuning(i)
    tuning_hyperparam(j)

    epc_ft_model = ExPostCombination_FinetunedModel(k_fold=i)
    epc_ft_model.first_validation()
    epc_ft_model.reassembly_batches()
    del ExPostCombination_FinetunedModel

# -----------------------------------------------------------------
