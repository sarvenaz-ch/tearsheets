# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:18:11 2024

@author: Sarvenaz
"""
import numpy as np

# Evaluation
from ragas.metrics import *
from ragas.langchain import RagasEvaluatorChain # langchain chain wrapper to convert a ragas metric into a langchain
import matplotlib.pyplot as plt

def ragas_eval_qa(query, result, eval_metrics_list = [faithfulness, answer_relevancy, context_relevancy], viz = False):
    """
    Parameters
    ----------
    query : string, question.
    result : dictionary, result from qa_chain
    eval_metrics_list : List, list of Ragas metrics. The default is [faithfulness, answer_relevancy, context_relevancy].
    viz : Boolean, If true, the function generated and shows a bar plot of the metrics scores. The default is False.

    Returns
    -------
    result : dictionary, original `result` variable with metrics score added as new fields. 
    """

    # make eval chains
    eval_chains = {
        m.name: RagasEvaluatorChain(metric=m) 
        for m in eval_metrics_list
    }  
    
    for metric, eval_chain in eval_chains.items():
        metric_name = f'{metric}_score'
        result[metric_name]= round(eval_chain({'query':query,
                                                 'source_documents':result['source_documents'],
                                                 'result':result['result']})[metric_name]
                                   , 2)
    # Bar plot of the metrics 
    if viz:
        plot_metric = {k:result[k] for k in [metric.name + '_score' for metric in eval_metrics_list]}
        plot_metrics_with_values(plot_metric, title = 'RAGAS Metrics')
        
    return result


def plot_metrics_with_values(metrics_dict, title='RAG Metrics'):
    """
    Plots a bar chart for metrics contained in a dictionary and annotates the values on the bars.
    Args:
    metrics_dict (dict): A dictionary with metric names as keys and values as metric scores.
    title (str): The title of the plot.
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, values, color='skyblue')
    # Adding the values on top of the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01,  # x-position
                 bar.get_y() + bar.get_height() / 2,  # y-position
                 f'{width:.4f}',  # value
                 va='center')
    plt.xlabel('Score')
    plt.title(title)
    plt.xlim(0, 1)  # Setting the x-axis limit to be from 0 to 1
    plt.show()
    

def ragas_eval_qa_avg(eval_questions, qa,  eval_metrics_list = [faithfulness, answer_relevancy, context_relevancy], viz = False):
    """
    Parameters
    ----------
    eval_questions : List. List of questions to evaluate the model based on

    qa : Langchain RAG chain. `return_source_documents` must be True
        
    eval_metrics_list : List, List of RAGAS metrics. The default is [faithfulness, answer_relevancy, context_relevancy].
    viz : Boolean, generated a bar plto of average scores for each metric. The default is False.

    Returns
    -------
    eval_answers : list of dictionaries. Each component of the list is the result returned from qa chain per question
    metric_tracker : list, list of metric scores.

    """

    # list of answers to the questions, along with their source document for evaliation
    eval_answers = [{k:qa(q)[k] for k in ['query', 'result', 'source_documents']} for q in eval_questions]
    
    # make eval chains
    eval_chains = {
        m.name: RagasEvaluatorChain(metric=m) 
        for m in eval_metrics_list
    }   
    
    metric_tracker = {}
    for metric, eval_chain in eval_chains.items():
        metric_name = f'{metric}_score'
        metric_tracker[metric_name] = []
        for eval_answer in eval_answers:
            metric_tracker[metric_name].append(eval_chain({'query':eval_answer['query'], 'source_documents':eval_answer['source_docuemtns'], 'result':eval_answer['result']})[metric_name])
        metric_tracker[f'{metric_name}_avg'] = np.mean(metric_tracker[metric_name]) # Calculating mean over each metric for an overall assesment
    
    if viz:
        metric_avg = {k:metric_tracker[k] for k in [metric.name + '_score_avg' for metric in eval_metrics_list]}
        plot_metrics_with_values(metric_avg, title = 'Avg RAG Metrics')
    
    return eval_answers, metric_tracker    



if __name__ == '__main__':
    import ragas_evaluations as m
    
    m.plot_metrics_with_values({'faithfulness_score': 1.0,
                              'answer_relevancy_score': 0.9468557283238227,
                              'context_relevancy_score': 0.025}, "Base Retriever ragas Metrics")
    
    
    # Generating a set of sample questions and answers to test score aggregation
    eval_questions = [
        'Where does Jerry work?',
        'What industry or industries is Jerry involved in?',
        'Is Jerry a charitable person?',
        'Who are Jerrys family memebers?',
        ]
    