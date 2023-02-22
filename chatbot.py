from sqlitedict import SqliteDict
import openai
import os
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.DEBUG)


class ChatBot():
    def __init__(self, model="text-davinci-003", temperature=0.0, database_name="queries.sqlite", context_data_file = "data.txt"):
        self.model = model
        self.temp = temperature
        self.db = SqliteDict(database_name)
        if Path('bad_responses.json').is_file():
            self.bad_responses = pd.read_json('bad_responses.json')
        else:
            self.bad_responses = pd.DataFrame(columns=['question','context','response'])
        if not Path('info.json').is_file():
            self.info={}
        else:
            with open('info.json') as f:
                self.info = json.load(f)
        self.embed_model = 'text-embedding-ada-002' 
        self.context_data_file = context_data_file
        self.curr_ques = None
        self.curr_response = None
        self.maxwords = 50 #Can change as required

    def get_response(self, ques):
        if ques in self.db:
            return self.db[ques]
        else:
            new_ques = self.parse_prompt(ques.lower())
            response = openai.Completion.create(
                model=self.model,
                prompt=new_ques,
                temperature=self.temp,
                max_tokens=2000)
            self.curr_ques = ques
            self.curr_response = response.choices[0]['text']
            return self.curr_response
    
    def parse_prompt(self, prompt):
        init_prompt = """You are a polite and helpful chatbot that answers questions related to Sashank Tirumala. You will try to answer questions as truthfully as possible and say I don't know if you are not sure. If the question seems unrelated to Sashank then you will politely decline to answer.\n"""
        context = self.get_context(prompt)
        final_prompt = init_prompt + context + prompt
        final_prompt= final_prompt+"?" if final_prompt[-1]!="?" else final_prompt
        return final_prompt
    
    def get_context(self, prompt):
        prompt_embed = np.array(openai.Embedding.create(
                        model=self.embed_model,
                        input=prompt
                        )['data'][0]['embedding'])
        final_context = ''
        df = pd.DataFrame(columns=('context', 'similarity'))
        for key, value in self.info.items():
            sim = np.dot(prompt_embed, np.array(value['embeddings']['data'][0]['embedding']))
            df = df.append({'context': key, 'similarity': sim}, ignore_index=True)
        df = df.sort_values('similarity', ascending=False)
        word_counter = 0
        for row in df.iterrows():
            word_counter = word_counter + len(row[1]['context'].split(' '))
            final_context = final_context + row[1]['context']
            if word_counter> self.maxwords:
                break
        return final_context+"\n"

    def generate_embeddings(self):
        with open(self.context_data_file, 'r') as f:
            for line in f.readlines():
                context = line.strip("\n")
                if context in self.info:
                    continue
                self.info[context] = {}
                self.info[context]['embeddings']= openai.Embedding.create(
                        model=self.embed_model,
                        input=context.lower()
                        )

        with open('info.json', 'w') as fp:
            json.dump(self.info, fp)
        
        logging.debug("Completed generating embeddings and wrote to info.json")
        


    def commit_response(self):
        self.db[self.curr_ques]=self.curr_response
        self.db.commit()

    def commit_negative_response(self):
        res={}
        res['question'] = self.curr_ques
        res['context'] = self.get_context(self.curr_ques.lower())
        res['response'] = self.curr_response
        self.bad_responses = self.bad_responses.append(res, ignore_index=True)
        self.bad_responses.to_json('bad_responses.json')
    
    def convert_bad_responses_to_string(self, filename):
        res=''
        for index, row in self.bad_responses.iterrows():
            res = res+"Q: "+row['question']+'\n'
            res = res+"C: "+row['context']+"\n"
            res = res+"A: "+row['response']+"\n\n"
        with open(filename, 'a') as f:
            f.write(res)


if __name__ == "__main__":
    ch = ChatBot()
    breakpoint()
    ch.get_response("Which all locations did Sashank intern in?")
    ch.generate_embeddings()
    pass


