import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
'''
Goemotions=dict()
Goemotions['info']="Results of BERT trained on Goemotions Data "
Goemotions['zero_shot']=dict()
Goemotions['zero_shot']=dict(
{
    'dev_set_accuracies':[
        0.49303621169916434,
        0.5125348189415042,
        0.5069637883008357,
        0.5376044568245125,
        0.4958217270194986,
        0.5097493036211699,
        0.4818941504178273,
        0.520891364902507,
        0.49303621169916434,
        0.5153203342618384,
        0.4986072423398329,
        0.38997214484679665,
        0.48467966573816157,
        0.49303621169916434,
        0.5097493036211699,
        0.5013927576601671,
        0.4735376044568245,
        0.5153203342618384,
        0.5013927576601671,
        0.5125348189415042,
        0.43175487465181056,
        0.45125348189415043,
        0.520891364902507,
        0.5125348189415042,
        0.5710306406685237,
        0.4568245125348189,
        0.4958217270194986
    ],
   'dev_set_macro_f1':[
        0.37153457273320284,
        0.35995423340961097,
        0.3622659280339474,
        0.3721727793209002,
        0.3414439088518843,
        0.34623493975903613,
        0.350743006993007,
        0.36270878468097006,
        0.35703983764126546,
        0.3476512907321202,
        0.34706600446568314,
        0.3421954174928398,
        0.35152883024571585,
        0.34707446808510634,
        0.35928614640048395,
        0.3547673531655225,
        0.3424051178219102,
        0.35310888136686014,
        0.3505279628612582,
        0.3530717609800308,
        0.365112804422531,
        0.36169949213427477,
        0.39286530885042914,
        0.36756659397149316,
        0.3824107311543548,
        0.35453648915187375,
        0.35343791321072243
  ]
}
)


Goemotions['frozen_bert']=dict(
{
    'accuracies': [
        0.4975,
        0.4975,
        0.5075,
        0.5075,
        0.5025,
        0.6406685236768802,
        0.6518105849582173,
        0.6545961002785515,
        0.6573816155988857,
        0.6601671309192201
    ],
    'macro_f1': [
        0.7517728049138184,
        0.7440119498943027,
        0.7498201470407608,
        0.7498574248738759,
        0.7469217671442141,
        0.8155387851045404,
        0.8195170702978603,
        0.8205238299699442,
        0.8201191375115362,
        0.821078431372549
    ]

}
)
'''
import argparse




def plot_zero_shot():
    #Zeroshot data
    global n
    output_dir = 'ckpt/twitter_zeroshot/bert-base-cased-goemotions-twitterzeroshot'
    output_eval_files=os.listdir(output_dir)
    global_steps=[]
    results=[]
    for output_eval_file in output_eval_files:
        if 'test' not in output_eval_file:
            continue
        global_steps.append(int(output_eval_file[5:-4]))
        with open(os.path.join(output_dir,output_eval_file),'r') as f:
            metrics=f.readlines()
            result=dict()
            for metric in metrics:
                metric,value=metric.strip().replace(" ","").split('=')
                result[metric]=float(value)
            results.append(result)

    accuracies=[results[i]['accuracy'] for i in range(len(results))]
    macro_f1s=[results[i]['macro_f1'] for i in range(len(results))]
    global_steps,accuracies,macro_f1s = zip(*sorted(zip(global_steps,accuracies,macro_f1s)))

    fig1 = plt.figure(n)
    n+=1
    plt.suptitle('Zero Shot Test Accuracies')
    plt.ylabel='Accuracy'
    plt.xlabel='Training Step Checkpoint'
    plt.plot(global_steps,accuracies)
    plt.autoscale(enable=True)
    plt.savefig('zero_shot_test_accuracies')
    fig2 = plt.figure(n)
    n+=1
    plt.suptitle('Zero Shot Test Macro_F1 scores')
    plt.ylabel='Macro F1 Score'
    plt.xlabel='Training Step Checkpoint'
    plt.plot(global_steps,macro_f1s)
    plt.autoscale(enable=True)
    plt.savefig('zero_shot_test_f1')

def plot_frozenbert_data():
    global n
    output_dir="ckpt/twitter_frozenbert/bert-base-cased-goemotions-twitter-frozenbert"
    output_dir_dev=os.path.join(output_dir,'dev')
    output_dir_test=os.path.join(output_dir,'test')
    if not os.path.isdir(output_dir_dev):
        print("dev folder does not exist!!")
        return
    if not os.path.isdir(output_dir_test):
        print("test folder does not exist!!")
        return
    dev_output_eval_files=os.listdir(output_dir_dev)
    test_output_eval_files=os.listdir(output_dir_test)

    for output_eval_files,mode in [(dev_output_eval_files,'dev'),(test_output_eval_files,'test')]:
        global_steps=[]
        results=[]
        for output_eval_file in output_eval_files:
            if mode not in output_eval_file:
                continue
            global_steps.append(int(output_eval_file[len(mode)+1:-4]))
            with open(os.path.join(output_dir,mode,output_eval_file),'r') as f:
                metrics=f.readlines()
                result=dict()
                for metric in metrics:
                    metric,value=metric.strip().replace(" ","").split('=')
                    result[metric]=float(value)
                results.append(result)

        accuracies=[results[i]['accuracy'] for i in range(len(results))]
        macro_f1s=[results[i]['macro_f1'] for i in range(len(results))]
        global_steps,accuracies,macro_f1s = zip(*sorted(zip(global_steps,accuracies,macro_f1s)))

        fig1 = plt.figure(n)
        plt.suptitle('FrozenBert '+mode+' Accuracies')
        plt.ylabel='Accuracy'
        plt.xlabel='Training Step Checkpoint'
        plt.plot(global_steps,accuracies)
        plt.autoscale(enable=True)
        plt.savefig('frozen_bert_'+mode+'_accuracies')
        n+=1
        fig2 = plt.figure(n)
        plt.suptitle('FrozenBert '+mode+' Macro_F1 scores')
        plt.ylabel='Macro F1 score'
        plt.xlabel='Training Step Checkpoint'
        plt.plot(global_steps,macro_f1s)
        plt.autoscale(enable=True)
        plt.savefig('frozen_bert_'+mode+'_f1')
        n+=1

def plot_data(output_dir='ckpt/group/bert-base-cased-goemotions-group',taxonomy='group'):
    global n
    #output_dir="ckpt/twitter_frozenbert/bert-base-cased-goemotions-twitter-frozenbert"
    output_dir_dev=os.path.join(output_dir,'dev')
    output_dir_test=os.path.join(output_dir,'test')
    if not os.path.isdir(output_dir_dev):
        print("dev folder does not exist!!")
        return
    if not os.path.isdir(output_dir_test):
        print("test folder does not exist!!")
        return
    dev_output_eval_files=os.listdir(output_dir_dev)
    test_output_eval_files=os.listdir(output_dir_test)

    for output_eval_files,mode in [(dev_output_eval_files,'dev'),(test_output_eval_files,'test')]:
        global_steps=[]
        results=[]
        for output_eval_file in output_eval_files:
            if mode not in output_eval_file:
                continue
            global_steps.append(int(output_eval_file[len(mode)+1:-4]))
            with open(os.path.join(output_dir,mode,output_eval_file),'r') as f:
                metrics=f.readlines()
                result=dict()
                for metric in metrics:
                    metric,value=metric.strip().replace(" ","").split('=')
                    result[metric]=float(value)
                results.append(result)

        accuracies=[results[i]['accuracy'] for i in range(len(results))]
        macro_f1s=[results[i]['macro_f1'] for i in range(len(results))]
        global_steps,accuracies,macro_f1s = zip(*sorted(zip(global_steps,accuracies,macro_f1s)))

        fig1 = plt.figure(n)
        plt.suptitle(taxonomy+' '+mode+' Accuracies')
        plt.ylabel='Accuracy'
        plt.xlabel='Training Step Checkpoint'
        plt.plot(global_steps,accuracies)
        plt.autoscale(enable=True)
        plt.savefig(taxonomy+'_'+mode+'_accuracies')
        n+=1
        fig2 = plt.figure(n)
        plt.suptitle(taxonomy+' '+mode+' Macro_F1 scores')
        plt.ylabel='Macro F1 score'
        plt.xlabel='Training Step Checkpoint'
        plt.plot(global_steps,macro_f1s)
        plt.autoscale(enable=True)
        plt.savefig(taxonomy+'_'+mode+'_f1')
        n+=1
#output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
n=1

#first get args
cli_parser = argparse.ArgumentParser()

cli_parser.add_argument("--out_dir", type=str, required=False, default=None, help="results dir")
cli_parser.add_argument("--taxonomy", type=str, required=True, default=None,help="taxonomy")
args=cli_parser.parse_args()
print("taxonomy:",args.taxonomy)
print("outputs dir: ",args.out_dir)

plot_data(args.out_dir,args.taxonomy)
#plot_zero_shot()
#plot_frozenbert_data()
print(n-1," figures plotted")