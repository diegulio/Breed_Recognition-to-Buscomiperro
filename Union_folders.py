#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:37:48 2021

@author: diegulio
"""

import shutil
import os 
import pandas as pd

# Solo se hace con gatos ya que solo se tomaron 100 aleatorios de esto, los otros son simple copiar y pegar
cats= pd.read_csv('cat_labels.csv')

ids = cats['id']

for id_ in ids:
    shutil.copy( 'gatos/'+id_ + '.jpg', 'features/'+ id_ + '.jpg')
