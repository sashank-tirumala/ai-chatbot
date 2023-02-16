from sqlitedict import SqliteDict
import openai
import os
import json
import pandas as pd
import numpy as np
class ChatBot():
    def __init__(self, model="text-davinci-003", temperature=0.0, database_name="queries.sqlite"):
        self.model = model
        self.temp = temperature
        self.db = SqliteDict(database_name)
        with open('info.json') as f:
            self.info = json.load(f)
        self.embed_model = 'text-embedding-ada-002' 
        self.curr_ques = None
        self.curr_response = None

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
        final_context = 0
        max_similarity = -2
        for key, value in self.info.items():
            sim = np.dot(prompt_embed, np.array(value['embeddings']['data'][0]['embedding']))
            if max_similarity<sim:
                max_similarity=sim
                final_context = value['context']
        final_context=final_context.replace('\n', '')
        return final_context+"\n"

    def generate_embeddings(self):
        for key, value in self.info.items():
            if value["embeddings"]==0:
                value["embeddings"] = openai.Embedding.create(
                        model=self.embed_model,
                        input=value["context"]
                        )
        with open('info.json', 'w') as fp:
            json.dump(self.info, fp)
    
    def commit_response(self):
        self.db[self.curr_ques]=self.curr_response
        self.db.commit()

    

if __name__ == "__main__":
    ch = ChatBot()
    breakpoint()
    ch.get_response()
    pass


