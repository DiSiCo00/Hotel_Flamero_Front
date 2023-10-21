import pandas as pd
import os

def html_sheader(header:str):
    return f"""<div class="comment_header" id="myHeader">
                        <h2>{header}</h2>
                        </div>"""


def html_score_badges(num):
    return f"""
    <br>
    <span class="badge">{str(num)}</span>
    """

def html_score_little_badges(num):
    return f"""
    <br>
    <span class="little_badge">{str(num)}/10</span>
    """


def html_comments(raiting, user, comment):
    
        return f"""<div class="comment_container">
        <div class="container_comments_header">
        <span class="little_badge">{str(raiting)}/10</span>
        <h5>{user}</h5>
        </div>
        <p>{comment}</p>
        </div>"""


def comment_section():
    html_code = str()
    data_comments = pd.read_csv("_Data/comments.csv")
    for raiting, comment, user in zip(data_comments["Score"],data_comments["Comentario_Positivo"], data_comments["Usuario"]):
        html_code += html_comments(raiting, user, comment)
    return html_code