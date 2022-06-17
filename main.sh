


# upstream
python3 main.py \
    --model_save_path "save_model/roberta_xlm_binary_ep" \
    --upstream true \
    --bert_model_name "xlm-roberta-large" \
    --submit_save_path "save_submit/your_result3"



# downstream
python3 main.py \
    --model_save_path 'save_model/roberta_xlm_binary_ep_cr' \
    --upstream false \
    --bert_model_name 'save_model/roberta_xlm_binary_ep' \
    --submit_save_path 'save_submit/your_result4'

