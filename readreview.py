# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:50:52 2019

@author: hp
"""

import csv
flag = 1

txt_file = r"productInfoXML-reviewed-mProducts.txt"
csv_file = r"fourth.csv"


with open(txt_file, "r") as in_text:
    in_reader = csv.reader(in_text, delimiter = '\t')
    with open(csv_file, "w") as out_csv:
        out_writer = csv.writer(out_csv)
        for row in in_reader:
            flag += 1 #nandain datanya
            #if (flag>=2018 and flag<=145600) :
            if (flag>=1 and flag<=5000) :
                flag += 1
                out_writer.writerow(row)
                print (flag)
            #if (flag>145600) : break
            if (flag>5000) : break
            #else : continue
                