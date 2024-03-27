# Define METRICS and UTILITIES:
import re
from collections import Counter
import numpy as np
from bert_score import score
from rouge_score import rouge_scorer
from tqdm import tqdm
from configuration import Config
from trie import MarisaTrie
config = Config()

def compute_f1(prediction, truth, tokenizer = None):
    if tokenizer is None:
      pred_tokens = prediction.split()
      truth_tokens = truth.split()
    else:
      pred_tokens = tokenizer.tokenize(prediction)
      truth_tokens = tokenizer.tokenize(truth)
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    # if no answer, f1=1 if they agree and 0 otherwise
    if len(truth_tokens) == 0 or len(pred_tokens) == 0:
      return int(truth_tokens == pred_tokens)
    # if there are no common tokens then f1 = 0
    if num_same == 0:
        return 0
    prec = 1.0 * num_same/len(pred_tokens)
    rec = 1.0 * num_same/len(truth_tokens)
    return (2 * prec * rec) / (prec + rec)

# Functions to retrieve from input text the options and the context chunks that are going to populate the trie.
def get_options(input_text):
    return [s.split(". Domanda:")[0].strip() for s in re.split("[0-9]\)",input_text.split("Opzioni:")[1].strip()) if len(s)>1]

def get_context_chunks(input_text):
    context_words = input_text.split(". Domanda: ")[0].split("Opzioni: ")[0].strip().split(" ")
    context_chunks = []
    for i in range(len(context_words)):
        context_chunks.append(" ".join(context_words[i:len(context_words)]))
    return context_chunks

def restrict_decode_vocab_trie(batch_idx,prefix_beam,choice_trie,attach_eos=False, min_len=1):
  if attach_eos and len(prefix_beam.tolist())>min_len:
    return choice_trie.get(prefix_beam.tolist())+[1]
  else:
    return choice_trie.get(prefix_beam.tolist())

#Example attributes:
#   - type: multichoice|factual|free-text
#   - input text: i.e., the input of the seq2seq model
#   - output text: i.e, the gold label
def _get_single_pred(example, model, tokenizer, constrained, model_dev, max_length,opt = []):
  #CASE 1 = MULTICHOICE (OPTIONS)
  if example["type"] == "multichoice":
    options = get_options(example["input_text"])
    if constrained:
      trie = MarisaTrie([[0]+tokenizer.encode(option) for option in options])
      generated_output = model.generate(tokenizer.encode(example["input_text"], return_tensors="pt").to(model_dev), max_length=max_length,
                                        prefix_allowed_tokens_fn=lambda batch_id,prefix:restrict_decode_vocab_trie(batch_id,prefix,trie))
      del trie
    else:
      generated_output = model.generate(tokenizer.encode(example["input_text"], return_tensors = "pt").to(model_dev), max_length = max_length)
    pred = tokenizer.decode(generated_output[0], skip_special_tokens = True)
    format_correct = pred in options
  #CASE 2 = FACTUAL (NO OPTIONS)
  elif example["type"]=="factual":
    if constrained:
      trie = MarisaTrie([[0]+tokenizer.encode(chunk) for chunk in get_context_chunks(example["input_text"])])
      generated_output = model.generate(tokenizer.encode(example["input_text"], return_tensors="pt").to(model_dev), max_length=max_length,
                                        prefix_allowed_tokens_fn=lambda batch_id,prefix:restrict_decode_vocab_trie(batch_id,prefix,trie,attach_eos=True, min_len=2))
      del trie
    else:
      generated_output = model.generate(tokenizer.encode(example["input_text"], return_tensors = "pt").to(model_dev), max_length = max_length)
    pred = tokenizer.decode(generated_output[0], skip_special_tokens = True)
    p = pred.split(" ")
    if len(p)==2 and p[1] == "mm":
      try:
       int(p[0])
       format_correct = True
      except: format_correct = False
    else: format_correct = False
    #CASE 3 = FREE-TEXT (NO OPTIONS)
  elif example["type"]=="free-text":
    ###### Generation config
    if len(opt)==3: rp,nb,lp = opt
    else: rp=1; nb=1; lp=1

    generated_output = model.generate(tokenizer.encode(example["input_text"], return_tensors = "pt").to(model_dev), max_length = max_length,repetition_penalty=rp,num_beams=nb,length_penalty=lp)
    pred = tokenizer.decode(generated_output[0], skip_special_tokens = True)
    format_correct = True
  return pred, example["output_text"], format_correct

def validate(model, tokenizer, dataset, constrained = True, verbose = False, opt = []):
    model_dev = model.device.type
    predictions = []
    references = []
    confidences = []
    format_correct = []

    # predict for each example
    for example in tqdm(dataset, leave = True, disable = True):
      pred,ref,f_correct = _get_single_pred(example, model, tokenizer, constrained = constrained, model_dev = model_dev, max_length = config.MAX_LEN,opt=opt)

      # confidence
      conf = compute_f1(pred,ref,tokenizer)
      confidences.append(conf)

      references.append(ref)
      predictions.append(pred)
      format_correct.append(f_correct)

    # calculate metrics for each type and overall
    question_types = list(set(dataset["type"]))
    metrics = {q:{} for q in question_types + ["overall"]}

    exact_match = [i==j for i,j in zip(predictions,references)]
    metrics["overall"]["strict_acc"] = round(100*len(np.where(exact_match)[0])/len(exact_match),2)
    metrics["overall"]["f1"] = round(100*np.mean([compute_f1(i,j,tokenizer) for i,j in zip(predictions,references)]),2)
    metrics["overall"]["format_acc"] = round(100*len(np.where(format_correct)[0])/len(format_correct),2)
    metrics["overall"]["size"] = len(references)

    for question_type in question_types:
      idx = list(np.where([t==question_type for t in dataset["type"]])[0])
      p = [predictions[i] for i in idx]
      r = [references[i] for i in idx]
      f = [format_correct[i] for i in idx]
      exact_match = [i==j for i,j in zip(p,r)]
      metrics[question_type]["strict_acc"] = round(100*len(np.where(exact_match)[0])/len(exact_match),2)
      metrics[question_type]["f1"] = round(100*np.mean([compute_f1(i,j,tokenizer) for i,j in zip(p,r)]),2)
      metrics[question_type]["format_acc"] = round(100*len(np.where(f)[0])/len(f),2)
      metrics[question_type]["size"] = len(r)

    return {
        'predictions': predictions,
        'references': references,
        'confidences': confidences,
        'metric': metrics
    }

def preprocess_function(examples):
    model_inputs = config.TOKENIZER(
        examples["input_text"],
        add_special_tokens=True,
        max_length=config.MAX_INPUT_TOKENS,
        pad_to_max_length=True,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )
    labels = config.TOKENIZER(
        examples["output_text"],
        add_special_tokens=True,
        max_length=config.MAX_OUTPUT_TOKENS,
        pad_to_max_length=True,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def return_line(unconst_res,const_res):
  line = {
        "overall":{
          "unconstrained":{
              "F1":unconst_res["metric"]["overall"]["f1"],
              "SA":unconst_res["metric"]["overall"]["strict_acc"],
              "FA":unconst_res["metric"]["overall"]["format_acc"],
          },
          "constrained":{
              "F1":const_res["metric"]["overall"]["f1"],
              "SA":const_res["metric"]["overall"]["strict_acc"],
              "FA":const_res["metric"]["overall"]["format_acc"],
          }
        },
        "factual":{
          "unconstrained":{
              "F1":unconst_res["metric"]["factual"]["f1"],
              "SA":unconst_res["metric"]["factual"]["strict_acc"],
              "FA":unconst_res["metric"]["factual"]["format_acc"]
          },
          "constrained":{
              "F1":const_res["metric"]["factual"]["f1"],
              "SA":const_res["metric"]["factual"]["strict_acc"],
              "FA":const_res["metric"]["factual"]["format_acc"],
          }
        },
        "free-text":{
          "unconstrained":{
              "F1":unconst_res["metric"]["free-text"]["f1"],
              "SA":unconst_res["metric"]["free-text"]["strict_acc"],
              "FA":unconst_res["metric"]["free-text"]["format_acc"]
          },
          "constrained":{
              "F1":const_res["metric"]["free-text"]["f1"],
              "SA":const_res["metric"]["free-text"]["strict_acc"],
              "FA":const_res["metric"]["free-text"]["format_acc"],
          }
        },
        "multichoice":{
          "unconstrained":{
              "F1":unconst_res["metric"]["multichoice"]["f1"],
              "SA":unconst_res["metric"]["multichoice"]["strict_acc"],
              "FA":unconst_res["metric"]["multichoice"]["format_acc"]
          },
          "constrained":{
              "F1":const_res["metric"]["multichoice"]["f1"],
              "SA":const_res["metric"]["multichoice"]["strict_acc"],
              "FA":const_res["metric"]["multichoice"]["format_acc"],
          }
        }
  }
  return line

def calculate_metrics_batch(dataset, tokenizer, model):
    predictions = list(dataset["best_pred"])
    references = list(dataset["ref"])

    format_correct = []
    for data in dataset.index:
      data_row = dataset.iloc[data,:]
      if  data_row["type"] == "multichoice":
        options = get_options(data_row["input_text"])
        options = [x.lower() for x in options]
        f_correct = str(data_row["best_pred"]).lower() in options
      elif data_row["type"] =="free-text":
        f_correct = True
      elif data_row["type"] =="factual":
        p = str(data_row["best_pred"]).split(" ")
        if len(p) == 2 and p[1] == "mm": f_correct  = True
        else: f_correct = False
      format_correct.append(f_correct)

    # calculate metrics for each type and overall
    question_types = list(set(dataset["type"]))
    metrics = {q:{} for q in question_types + ["overall"]}

    exact_match = [i==j for i,j in zip(predictions,references)]
    metrics["overall"]["strict_acc"] = round(100*len(np.where(exact_match)[0])/len(exact_match),2)
    metrics["overall"]["f1"] = round(100*np.mean([compute_f1(str(i),str(j),tokenizer) for i,j in zip(predictions,references)]),2)
    metrics["overall"]["format_acc"] = round(100*len(np.where(format_correct)[0])/len(format_correct),2)
    metrics["overall"]["size"] = len(references)

    for question_type in question_types:
      idx = list(np.where([t==question_type for t in dataset["type"]])[0])
      p = [predictions[i] for i in idx]
      r = [references[i] for i in idx]
      f = [format_correct[i] for i in idx]
      exact_match = [i==j for i,j in zip(p,r)]
      metrics[question_type]["strict_acc"] = round(100*len(np.where(exact_match)[0])/len(exact_match),2)
      metrics[question_type]["f1"] = round(100*np.mean([compute_f1(str(i),str(j),tokenizer) for i,j in zip(p,r)]),2)
      metrics[question_type]["format_acc"] = round(100*len(np.where(f)[0])/len(f),2)
      metrics[question_type]["size"] = len(r)

    return {'predictions': predictions,'references': references,'metric': metrics}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = config.TOKENIZER.batch_decode(predictions, skip_special_tokens = True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, config.TOKENIZER.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = config.TOKENIZER.batch_decode(labels, skip_special_tokens = True)
    exact_match = [i==j for i,j in zip(decoded_preds,decoded_labels)]
    accuracy = len(np.where(exact_match)[0])/len(exact_match)
    f1 = np.mean([compute_f1(decoded_preds[i],decoded_labels[i],config.TOKENIZER) for i in range(len(decoded_preds))])

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for i in np.arange(len(decoded_preds)):
        scores.append(scorer.score(decoded_preds[i], decoded_labels[i])["rougeL"].fmeasure)
    rouge = np.mean(scores)

    model_type = "IVN-RIN/bioBIT"

    bitscore5 = score(cands=decoded_preds,refs=decoded_labels,model_type=model_type,num_layers=5, verbose=False)[2][0].item()
    bitscore7 = score(cands=decoded_preds, refs=decoded_labels, model_type=model_type, num_layers=7, verbose=False)[2][0].item()
    bitscore9 = score(cands=decoded_preds, refs=decoded_labels, model_type=model_type, num_layers=9, verbose=False)[2][0].item()
    bitscore12 = score(cands=decoded_preds, refs=decoded_labels, model_type=model_type, num_layers=12, verbose=False)[2][0].item()

    return {"accuracy": accuracy, "f1":f1, "rouge":rouge,
            "BertScore5":bitscore5,"BertScore7":bitscore7,
            "BertScore9":bitscore9,"BertScore12":bitscore12}


