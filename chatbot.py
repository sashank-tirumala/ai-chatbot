from sqlitedict import SqliteDict
import openai
import os
class ChatBot():
    def __init__(self, model="text-davinci-003", temperature=0.0, database_name="queries.sqlite"):
        self.model = model
        self.temp = temperature
        self.db = SqliteDict(database_name)

    def get_response(self, ques):
        ques = self.parse_prompt(ques)
        if ques in self.db:
            return self.db[ques]
        else:
            response = openai.Completion.create(
                model=self.model,
                prompt=ques,
                temperature=self.temp)
            self.db[ques]=response.choices[0]['text']
            self.db.commit()
            return response.choices[0]['text']
    
    def parse_prompt(self, prompt):
        return prompt.lower()

    

if __name__ == "__main__":
    openai.api_key = "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"
    ch = ChatBot()
    breakpoint()
    ch.get_response()
    pass


