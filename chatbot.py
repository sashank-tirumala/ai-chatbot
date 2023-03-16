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
        if Path('good_responses.json').is_file():
            self.good_responses = pd.read_json('good_responses.json')
        else:
            self.good_responses = pd.DataFrame(columns=['question','context','response'])
        self.embed_model = 'text-embedding-ada-002' 
        self.context_data_file = context_data_file
        self.curr_ques = None
        self.curr_response = None
        self.msg_history = [{"role": "system", "content": "You are a helpful assistant."}]
        init_prompt = self.parse_prompt("")
        self.msg_history = self.msg_history + [{"role":"user", "content":init_prompt}]

    def get_response(self, ques):
        self.curr_ques = ques
        new_ques = [{"role":"user", "content":ques}]
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                    messages= self.msg_history+new_ques)

        self.curr_response = response.choices[0]['message']['content']
        self.update_msg_history(ques, self.curr_response)

        return self.curr_response
    
    def update_msg_history(self, ques, response):
        """
        For now allows a max chat history of 10x
        """
        if len(self.msg_history)>10: 
            self.msg_history.pop(2) 
            self.msg_history.pop(3) 
        self.msg_history.append({"role":"user", "content":ques})
        self.msg_history.append({"role":"assistant", "content":response})
        print(len(self.msg_history))
        print(self.msg_history)

    def parse_prompt(self, prompt):
        info = [        
        "Sashank Tirumala is a final year masters student of robotics sciences at Carnegie Mellon University. He will graduate in August 2023. He has experience in Deep Learning, Computer Vision, Robotics and Reinforcement Learning.",
        "He interned in Tesla Autopilot from 2022 June to Aug 31st 2022. There he worked on developing a traffic sign pose estimator using deep learning.",
        "Specifically he worked on all stacks, curating the ground truth dataset,setting up the metrics to training the transformer model on the dataset",
        "and finally pushed his model onto V10.69 update in Tesla Autopilot. He trained this model on over 800 A100 GPU's. Apart from this, he also worked on developing a novel neural network architecture",
        "that could get detect objects at 4X the distance without regressing on detecting nearby objects in Tesla.",
        "Sashank also did research in robot manipulation with David Held in Grad school in CMU from 2021-2023. He worked on using tactile sensing and deep learning",
        "to increase the accuracy of cloth grasping. A method he developed improved accuracy over vision based cloth grasping by 80%.",
        "This led to a publication in IROS 2022, \"Learning to singulate layers of cloth with tactile sensing\" that won the best paper award in Deformable Object Manipulation Category.",
        "Currently Sashank is researching implicit representations for cloth manipulation with Dave. Prior to this Sashank did his undergraduate degree in IIT Madras from 2016 to 2021 in robotics.",
        "During this time he interned for 6 months in Robert Bosch Center for Cyberphysical Systems in Dec - August 2020. Here he developed reinforcement learning algorithms to make a quadruped robot",
        "walk that led to 2 publications in Conference on Robot Learning and Robot Human Interactions. Specifically he developed a novel neural network architecture that required ten times",
        "lesser demonstrations to learn quadruped motion and worked on linear reinforcement learning controllers to enable a quadruped robot to walk on slopes.",
        "Sashank is very excited about Large Language Models, his initial foray into this is this chatbot that he developed here. He has another exciting project in this space that is coming soon!",
        "Sashank requires a H1b visa to work in US, he will have a 3 year OPT period during which he can work without H1b.",
        "Sashank in grad school has done courses in Deep Learning, Machine Learning, Reinforcement Learning and Computer Vision. Sashank lives in Pittsburgh USA,",
        "and is willing to relocate within USA for a job, and in India he lived in Bangalore, Hyderabad and Chennai.\n",   
        ]
        init_prompt = "You are a polite and helpful chatbot that answers questions related to Sashank Tirumala to potential job recruiters. You will answer questions honestly based on the information I provide below while also painting Sashank in a positive light. If the question seems unrelated to Sashank (Example: how many planets in the solar system), answer that that seems unrelated to Sashank. If the question is not present in the information below, (Example: Who is Sashank's girlfriend), answer that you were not trained on data pertaining to that question.\n"
        info = "".join(info)
        return init_prompt + info + prompt


    def commit_response(self):
        res={}
        res['question'] = self.curr_ques
        res['response'] = self.curr_response
        self.good_responses = self.good_responses.append(res, ignore_index=True)
        self.good_responses.to_json('good_responses.json')


    def commit_negative_response(self):
        res={}
        res['question'] = self.curr_ques
        res['response'] = self.curr_response
        self.bad_responses = self.bad_responses.append(res, ignore_index=True)
        self.bad_responses.to_json('bad_responses.json')
    
    def convert_responses_to_string(self, filename, response_type='bad'):
        res=''
        resp = self.bad_responses if response_type=='bad' else self.good_responses
        filename= 'bad_data.txt' if response_type=='bad' else 'good_data.txt'
        for index, row in resp.iterrows():
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


