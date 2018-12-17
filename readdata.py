# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:44:42 2018

@author: hp
"""

import csv
csv_record_counter = 1
flag = 1

txt_file = r"productinfo/productinfo.txt"
csv_file = r"productinfo.csv"


with open(txt_file, "r") as in_text:
    in_reader = csv.reader(in_text, delimiter = '\t')
    with open(csv_file, "w") as out_csv:
        out_writer = csv.writer(out_csv)
        for row in in_reader:
            flag += 1
            if (flag>=1000&flag<=2000) :
                csv_record_counter += 1
                flag += 1
                out_writer.writerow(row)
                print ("flag")
                print (flag)
            if (flag>3000) : break
            #else : continue
                