import argparse
from curses.ascii import isdigit
import os
import csv

#split the 1600000 training examples between equally distributing the labels


def split(args,train,dev,test):
    label_dict=dict()
    with open(args.file,encoding='latin1') as f:
        for (i, line) in enumerate(f):
            line = line.strip()
            items = line.split("\t")
            #keep labels list of string
            labels = list(items[1].split(","))
            if (len(labels)>1):
                print("beware, more than 1 label for line {}".format(line))
            for label in labels:
                if label not in label_dict:
                    print('new label encountered: {}'.format(label))
                    if not label.isnumeric():
                        print(label,"line {} label is not numeric. Line: ".format(i),line,"items: ",items)
                        return
                    label_dict[label]=[]
                label_dict[label].append(items[0])
    print("Collected data")
    for l in label_dict:
        print('label {} has {} examples'.format(l,len(label_dict[l])))

    #splitting data 
    #get indices
    print('splitting train:dev:test in ratio {}:{}:{}'.format(train,dev,test))
    index_dict=dict()
    for l in label_dict:
        #number of examples
        num=len(label_dict[l])
        if (args.max_num is not None) and (args.max_num<num):
            num=args.max_num
        train_num=int(num*train)
        dev_index_start=train_num+1
        dev_num=int(num*dev)
        test_index_start=dev_index_start+dev_num+1
        print('label {}: train examples: 0 to {}, dev {} to {}, test {} to {}'.format(l,train_num,dev_index_start,test_index_start-1,test_index_start,num))
        index_dict[l]=[dev_index_start,test_index_start]

    #split data
    with open(args.train_file,'w',encoding='latin1') as ff:
        tsv_writer = csv.writer(ff, delimiter='\t')
        for label in index_dict:
            line_list=label_dict[label]
            dev_index_start,test_index_start=index_dict[label]
            for i in range(dev_index_start):
                tsv_writer.writerow([line_list[i],label])
    with open(args.dev_file,'w',encoding='latin1') as ff:
        tsv_writer = csv.writer(ff, delimiter='\t')
        for label in index_dict:
            line_list=label_dict[label]
            dev_index_start,test_index_start=index_dict[label]
            for i in range(dev_index_start,test_index_start):
                tsv_writer.writerow([line_list[i],label])   
    with open(args.test_file,'w',encoding='latin1') as ff:
        tsv_writer = csv.writer(ff, delimiter='\t')
        for label in index_dict:
            line_list=label_dict[label]
            dev_index_start,test_index_start=index_dict[label]
            num=len(line_list)
            if (args.max_num is not None) and (args.max_num<num):
                num=args.max_num
            for i in range(test_index_start,num):
                tsv_writer.writerow([line_list[i],label]) 

        

        

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--file", type=str, required=True, help="Name of the file to split")
    cli_parser.add_argument("--ratio", type=str, required=True, help="train_dev_test. Underscore separated integers please")
    cli_parser.add_argument("--train_file", type=str, required=False, default='train.tsv', help="Optional name for train file")
    cli_parser.add_argument("--test_file", type=str, required=False, default='test.tsv', help="Optional name for test file")
    cli_parser.add_argument("--dev_file", type=str, required=False, default='dev.tsv', help="Optional name for dev file")
    cli_parser.add_argument("--max_num", type=int, required=False, help="To make smaller files")

    args = cli_parser.parse_args()

    if not os.path.exists(args.file):
        print("file '{}' does not exist. Check the path?".format(args.file))
    
    train,dev,test=args.ratio.split('_')
    train=int(train)*100
    dev=int(dev)*100
    test=int(test)*100
    summ=train+dev+test
    train/=summ
    dev/=summ
    test/=summ
    
    split(args,train,dev,test)


