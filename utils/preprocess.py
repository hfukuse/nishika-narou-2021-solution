import re
import pandas as pd


def remove_url(sentence):
    ret = re.sub(r"(http?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", sentence)
    return ret

def processing_ncode(input_df: pd.DataFrame):#引用:https://www.nishika.com/competitions/21/topics/164
    output_df = input_df.copy()

    num_dict = {chr(i): i - 65 for i in range(65, 91)}

    def _processing(x, num_dict=num_dict):
        y = 0
        for i, c in enumerate(x[::-1]):
            num = num_dict[c]
            y += 26 ** i * num
        y *= 9999
        return y

    tmp_df = pd.DataFrame()
    tmp_df['_ncode_num'] = input_df['ncode'].map(lambda x: x[1:5]).astype(int)
    tmp_df['_ncode_chr'] = input_df['ncode'].map(lambda x: x[5:])
    tmp_df['_ncode_chr2num'] = tmp_df['_ncode_chr'].map(lambda x: _processing(x))

    output_df['ncode_num'] = tmp_df['_ncode_num'] + tmp_df['_ncode_chr2num']
    return output_df

def count_keyword(x):
    return x.keyword.count(" ")

def count_nn_story(x):
    return x.story.count("\n\n")

def count_n_story(x):
    return x.story.count("\n")