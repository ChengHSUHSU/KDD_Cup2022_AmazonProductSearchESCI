#!/bin/bash
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
#  
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

IDS=("1")
IDS+=("2")
IDS+=("3")

FILES=("task_1_query-product_ranking")
FILES+=("task_2_multiclass_product_classification")
FILES+=("???????????????")



for i in "${!IDS[@]}"
do  
    ID="${IDS[$i]}"
    #FILENAME="${FILES[$i]}"
    FOLDER_PATH="task${ID}"
    mkdir -p ${FOLDER_PATH}
    cd ${FOLDER_PATH}
    aicrowd dataset download --challenge esci-challenge-for-improving-product-search "Task ${ID}*" ${FOLDER_PATH}
    unzip "*.zip"
    #cp -r data/processed/public/${FILENAME} ../${FILENAME}
    cd ..
done 