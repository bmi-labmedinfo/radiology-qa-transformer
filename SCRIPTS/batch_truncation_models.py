# BATCH-TRUNCATION MODELS + VALIDATION
import re
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from configuration import Config
from dataset_creation import *
from metrics import *
config = Config()

class BatchTruncation_PretrainedModel:
    def __init__(self, k_fold=0, seed=config.SEEDS[0]):
        self.k = k_fold
        self.seed = seed
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_CHECKPOINT)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
        config.USE_PRETRAINED = True
        config.POST_PROCESSING = False
        self.res_dir = config.RESULTS_DIR + "batch-truncation/pretrained/"
        # DATASET PREPARATION
        self.multitask_dataset = bt_k_fold_splitting(self.k, self.seed)
        self.sample_test = self.multitask_dataset["test"].shuffle(seed)
        self.sample_test = self.sample_test.map(lambda example: {"output_text": example["output_text"].lower()})

    def validation(self):
        print("> BT_pretrained_model (fold",self.k,"): VALIDATION START")
        unconst_res = validate(self.model, self.tokenizer, self.sample_test, constrained=False, verbose=True)
        for r in unconst_res["metric"]:
            unconst_res["metric"][r]["fold"] = self.k
        unc_df = pd.DataFrame(unconst_res["metric"])
        print(unc_df)
        unc_df.to_csv(self.res_dir + "BT_PT_unc_results_" + str(self.k) + ".csv")
        print(f"BT_PT_unc_results {self.k} fold: SAVED!")

        const_res = validate(self.model, self.tokenizer, self.sample_test, constrained=True, verbose=True)
        for r in const_res["metric"]:
            const_res["metric"][r]["fold"] = self.k
        con_df = pd.DataFrame(const_res["metric"])
        print(con_df)
        con_df.to_csv(self.res_dir + "BT_PT_con_results_" + str(self.k) + ".csv")
        print(f"BT_PT_con_results {self.k} fold: SAVED!")

        pretrained_table = pd.DataFrame({"id": self.sample_test["id"],
                                         "type": self.sample_test["type"],
                                         "input_text": self.sample_test["input_text"],
                                         "reference": self.sample_test["output_text"],
                                         "unc_pred": unconst_res["predictions"],
                                         "con_pred": const_res["predictions"]})
        pretrained_table.to_csv(self.res_dir+ "BT_PT_table" + str(self.k) + ".csv")
        print(f"BT_PT_table {self.k} fold: SAVED!")
        print("> BT_pretrained_model (fold", self.k, "): VALIDATION END")

class BatchTruncation_FinetunedModel:
    def __init__(self, k_fold=0,seed=config.SEEDS[0]):
        self.k = k_fold
        self.seed = seed
        self.output_dir = config.OUTPUT_DIR + "batch-truncation/"
        self.ft_model = AutoModelForSeq2SeqLM.from_pretrained(self.output_dir + "output_" + str(self.k)+"_eval-loss")
        self.ft_tokenizer = AutoTokenizer.from_pretrained(self.output_dir + "output_" + str(self.k)+"_eval-loss")
        config.USE_PRETRAINED = False
        config.POST_PROCESSING = True
        self.res_dir = config.RESULTS_DIR + "batch-truncation/finetuned/"
        # DATASET PREPARATION
        self.multitask_dataset = bt_k_fold_splitting(self.k, self.seed)
        self.sample_test = self.multitask_dataset["test"]

    # Post-processing: free-text answers
    def pp_pred(self,pred):
        new_pred = []
        for i in np.arange(len(pred)):
            t = pred[i]
            resultantList = []
            for element in t.split(","):
                if element.strip() not in resultantList:
                    resultantList.append(element.strip())
            res = ""
            for n_el in np.arange(len(resultantList)):
                if n_el < len(resultantList) - 1:
                    # print(resultantList[n_el])
                    if resultantList[n_el] == "[not applicable]":
                        ""
                    else:
                        res = res + resultantList[n_el] + ", "
                else:
                    words = resultantList[n_el].split(" ")
                    es = re.search("[q|w|r|t|y|p|s|d|f|g|h|j|k|l|z|x|c|v|b|n|m|-]{1}$", words[-1])
                    if es: words.pop()
                    for v in np.arange(2):
                        if len(words) > 1 and len(words[-1]) <= 4:
                            words.pop()
                    if len(words) > 1 or (len(words) == 1 and len(words[0]) > 10):
                        for n_w in np.arange(len(words)):
                            res = res + words[n_w] + " "
            res = res.strip()
            if len(res)>1 and res[-1] == ",": res = res[:-1]
            new_pred.append(res)
        return new_pred

    # Calculating metrics after post-processing
    def calculate_metrics_pp(self,dataset, tokenizer, model):
        predictions = list(dataset["pred"])
        references = list(dataset["ref"])

        format_correct = []
        for data in dataset.index:
            data_row = dataset.iloc[data, :]
            if data_row["type"] == "multichoice":
                options = get_options(data_row["input_text"])
                options = [x.lower() for x in options]
                f_correct = str(data_row["pred"]).lower() in options
            elif data_row["type"] == "free-text":
                f_correct = True
            elif data_row["type"] == "factual":
                p = str(data_row["pred"]).split(" ")
                if len(p) == 2 and p[1] == "mm":
                    f_correct = True
                else:
                    f_correct = False
            format_correct.append(f_correct)

        # calculate metrics for each type and overall
        question_types = list(set(dataset["type"]))
        metrics = {q: {} for q in question_types + ["overall"]}

        exact_match = [i == j for i, j in zip(predictions, references)]
        metrics["overall"]["strict_acc"] = round(100 * len(np.where(exact_match)[0]) / len(exact_match), 2)
        metrics["overall"]["f1"] = round(
            100 * np.mean([compute_f1(str(i), str(j), tokenizer) for i, j in zip(predictions, references)]), 2)
        metrics["overall"]["format_acc"] = round(100 * len(np.where(format_correct)[0]) / len(format_correct), 2)
        metrics["overall"]["size"] = len(references)

        for question_type in question_types:
            idx = list(np.where([t == question_type for t in dataset["type"]])[0])
            p = [predictions[i] for i in idx]
            r = [references[i] for i in idx]
            f = [format_correct[i] for i in idx]
            exact_match = [i == j for i, j in zip(p, r)]
            metrics[question_type]["strict_acc"] = round(100 * len(np.where(exact_match)[0]) / len(exact_match), 2)
            metrics[question_type]["f1"] = round(
                100 * np.mean([compute_f1(str(i), str(j), tokenizer) for i, j in zip(p, r)]), 2)
            metrics[question_type]["format_acc"] = round(100 * len(np.where(f)[0]) / len(f), 2)
            metrics[question_type]["size"] = len(r)

        return {'predictions': predictions, 'references': references, 'metric': metrics}

    # Validation: including post-processing
    def validation(self):
        print("> BT_finetuned_model (fold", self.k, "): VALIDATION START")
        unc_results = validate(self.ft_model, self.ft_tokenizer, self.sample_test, constrained=False, opt=[])
        for r in unc_results["metric"]:
            unc_results["metric"][r]["fold"] = self.k
        unc_df = pd.DataFrame(unc_results["metric"])
        print(unc_df)
        unc_df.to_csv(self.res_dir + "BT_FT_unc_results_" + str(self.k) + ".csv")
        print(f"BT_FT_unc_results {self.k} fold: SAVED!")

        con_results = validate(self.ft_model, self.ft_tokenizer, self.sample_test, constrained=True, opt=[])
        for r in con_results["metric"]:
            con_results["metric"][r]["fold"] = self.k
        con_df = pd.DataFrame(con_results["metric"])
        print(con_df)
        con_df.to_csv(self.res_dir + "BT_FT_con_results_" + str(self.k) + ".csv")
        print(f"BT_FT_con_results {self.k} fold: SAVED!")

        prediction_table = pd.DataFrame({"id": self.sample_test["id"],
                                        "type": self.sample_test["type"],
                                        "input_text": self.sample_test["input_text"],
                                        "ref": unc_results["references"],
                                        "unc_pred": unc_results["predictions"],
                                        "con_pred": con_results["predictions"]})
        prediction_table.to_csv(self.res_dir + "BT_FT_table" + str(self.k) + ".csv")
        print(f"BT_FT_table {self.k} fold: SAVED!")

        #free-text post processing
        ft_all = prediction_table.loc[prediction_table["type"] == "free-text"]
        b = ft_all.loc[prediction_table["ref"] != "[not applicable]"]
        tab = b.iloc[:, [1, 4, 5]].reset_index()
        new_pred = self.pp_pred(tab["unc_pred"])
        idx = tab["index"]
        new_tab = prediction_table.copy()

        for i in np.arange(len(idx)):
            new_tab["unc_pred"].iloc[idx[i]] = new_pred[i]
            new_tab["con_pred"].iloc[idx[i]] = new_pred[i]
        new_tab.to_csv(self.res_dir + "PP_BT_FT_table_" + str(self.k) + ".csv")
        print(f"PP_BT_FT_table {self.k} fold: SAVED!")

        unc_sample_table = new_tab.iloc[:, 1:6].copy()
        unc_sample_table = unc_sample_table.rename(columns={"unc_pred": "pred"})
        unc_results = self.calculate_metrics_pp(unc_sample_table, self.ft_tokenizer, self.ft_model)

        con_sample_table = new_tab.iloc[:, 1:5].copy()
        con_sample_table["con_pred"] = new_tab["con_pred"].copy()
        con_sample_table = con_sample_table.rename(columns={"con_pred": "pred"})
        con_results = self.calculate_metrics_pp(con_sample_table, self.ft_tokenizer, self.ft_model)

        for r in unc_results["metric"]:
            unc_results["metric"][r]["fold"] = self.k
        unc_df = pd.DataFrame(unc_results["metric"])
        print(unc_df)
        unc_df.to_csv(self.res_dir + "BT_PP_unc_results_" + str(self.k) + ".csv")
        print("BT_PP_unc_results" + str(self.k) + "fold: SAVED!")

        for r in con_results["metric"]:
            con_results["metric"][r]["fold"] = self.k
        con_df = pd.DataFrame(con_results["metric"])
        print(con_df)
        con_df.to_csv(self.res_dir  + "BT_PP_con_results_" + str(self.k) + ".csv")
        print("BT_PP_con_results" + str(self.k) + "fold: SAVED!")