
# Import Initialization
import spacy
import pandas as pd
import csv
import torch
import numpy as np
from spacy.matcher import Matcher

from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction
from transformers import AutoModelWithLMHead, AutoTokenizer

BERT_MODEL = 'bert-large-uncased'
TEMPLATE = "twosentences" # "onesentence"

class BiasDetermination():
    def __init__(self):
        
        # Initialize the matcher with the shared vocab
        self._tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self._modelmask = BertForMaskedLM.from_pretrained(BERT_MODEL)
            
    def SkillWEAT(self, skillList):
        
        # Prior Probabilities: For all skills in the skill list, we will get the prior probabilities, aka the benchmark probabilities, once.
        preds, mask_index = self.MaskPredTask(
            skill=None, tokenizer=self._tokenizer, prior=True
        )
        priors = self.Pred2BiasProb(self._tokenizer, preds, mask_index)

        priorMan   = np.array(priors.loc[priors['Predicted']=='he','Prob'])
        priorWoman = np.array(priors.loc[priors['Predicted']=='she','Prob'])
        
        # Set up of the data frame in which we will store the output
        WEAToutput = pd.DataFrame(columns=["Skill", "WEATscore"])
        # For every skill in the skill-list, we acquire the WEAT Score.
        
        for i, sk in enumerate(skillList):
            # First we go from skill to mask predictions relevant to WEAT.
            
            preds, mask_index = self.MaskPredTask(
                sk, self._tokenizer, prior=False
            )
            # Then we convert the prediction to probabilities and collect the relevant ones for WEAT.
            output = self.Pred2BiasProb(self._tokenizer, preds, mask_index)
            
            predMan   = np.array(output.loc[output['Predicted']=='he','Prob'])
            predWoman = np.array(output.loc[output['Predicted']=='she','Prob'])
            
            # Calculate Weat Score from predictions and priors if the input was specified correctly
            weat_score = 0
            if (priorMan==0 or priorWoman == 0): 
                print(" Warning: Text does not fit into Template")
            else:
                log_bias_score = np.log(predMan / priorMan) - np.log(predWoman / priorWoman)
                weat_score = log_bias_score.item()
            
            WEAToutput = WEAToutput.append(
                {"Skill": sk, "WEATscore": weat_score},
                ignore_index=True,
            )
                
        return WEAToutput

    def MaskPredTask(self, skill,tokenizer, prior=False):
        """Go from skill provided, to text consisting of the WEAT template (with [MASK] included) 
        Afterwards, text is converted (I)to tokenized text with sentence_ids and finally (II)evaluated in the model (III) resulting to prediction of the masked index."""
        
        # Skill is put in the the template chosen, dependent on the Language and TEMPLATE type of one or two sentences.
        # If prior is False we get template for mask prediction related to the skill 
        # If prior is True we get template for mask prediction unrelated to a specific skill, that is the benchmark.
        # The benchmark allows you to see how the incorporation of the skill in the template affects the prediction.
                    
        text = ""
        
        if TEMPLATE=="onesentence":

            if (prior==False):
                text = f"[CLS] [MASK] can {skill} . [SEP]"
            else:
                text = f"[CLS] [MASK] can [MASK] . [SEP] "
            
            # Tokenize input and get masked_index
            tokenized_text = tokenizer.tokenize(text)
            masked_index = tokenized_text.index('[MASK]')
            # Get the corresponding ids for sentence 1 and 2.
            segments_ids = [0]*len(tokenized_text)
            
        elif TEMPLATE == "twosentences":
            
            if (prior==False):
                text = f"[CLS] [MASK] can {skill} . [SEP] [MASK] is suitable for the job . [SEP]"
            else:
                text = f"[CLS] [MASK] can [MASK] . [SEP] [MASK] is suitable for the job . [SEP]"

            print(text)
            # Tokenize input and get masked_index
            tokenized_text = tokenizer.tokenize(text)
            masked_index = tokenized_text.index('[MASK]')
            # Get the corresponding ids for sentence 1 and 2.
            sent1 = tokenized_text.index('[SEP]') + 1
            sent2 = len(tokenized_text)
            segments_ids = [0]*(sent1) + [1]*(sent2-sent1)
        else:
            print(f" Template specified is wrong , change variable TEMPLATE to \"onesentence\" or \"twosentences\", now it is {TEMPLATE}.")
        
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Evaluate the model once and then make all relevant predictions.
        self._modelmask.eval()

        # Predict all tokens
        with torch.no_grad():
            outputs = self._modelmask(
                tokens_tensor, token_type_ids=segments_tensors
            )
            predictions = outputs[0]

        return predictions, masked_index

    def Pred2BiasProb(self, tokenizer, preds, mask_index, top_k=300):
        """From Masked Prediction output this function first creates top k predictions and their associated probabilities via the method softmax of torch.nn.
        Then from this output of predicted words and probabilities, collects the relevant bias output for the gender dimension. """
        # First from preds to top k probs
        probs = torch.nn.functional.softmax(preds[0, mask_index], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        # Collect the top k probs in a readable output with their predicted words
        output = pd.DataFrame(
            [
                tokenizer.convert_ids_to_tokens([pred_idx])[0],
                top_k_weights[i].item(),
            ]
            for i, pred_idx in enumerate(top_k_indices)
        )
        output.columns = ["Predicted", "Prob"]
        # Select the relavant predicted words for the bias in the gender dimension.
        biasOutput = output.loc[output["Predicted"].isin(["she", "he"])]

        return biasOutput

# you can use main to test the code
def main():
    
    biasDeterminer = BiasDetermination()
    
    filename = 'ESCO_EN.csv' 
    #     filenames = ['ESCO_EN.csv','ONET_EN.csv']
    
    filePath = "data/" + filename
    # skillList is read where each row is a separate list to be evaluated, each row may be of different size
    with open(filePath, newline='') as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=';')
        data = [list(filter(None, row)) for row in reader]
    
    OUTPUT = pd.DataFrame(columns=['Code','MalePercentage', 'Skill', 'WEATscore'])
    for i, skillList in enumerate(data):
        print(skillList)
        output = biasDeterminer.SkillWEAT(skillList[2:]) # skillList[2:] because the first entry is the ONET/ESCO-Code, second male_percentage 
        print(skillList[0])
        occup_code  = skillList[0]  
        malep = skillList[1] 
        output.insert(0,'MalePercentage',malep, True)
        output.insert(0,'Code',occup_code, True)
        print(output)
        OUTPUT =  pd.concat([OUTPUT,output])
    fileOutputPath = f"data/OUTPUT_{filename}"        
    OUTPUT.to_csv(fileOutputPath, index = False, header=True, sep = ';')
if __name__ == "__main__":
    main()

