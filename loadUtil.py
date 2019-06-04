import pandas as pd
class loadUtil():
    @staticmethod
    def loadSampleCSV():
        df = pd.read_csv("./data/nlp/sample_data.txt", 
                    delimiter="\t",
                    names=['id','label','weight','feature'],
#                     chunksize=5,
#                     iterator=False,
                   )
        return df
