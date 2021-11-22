declare -a DirArray=("without_numbers" "without_symbols" "without_symbols_numbers")
declare -a TopicArray=("legal" "scientific")

for dir in ${DirArray[@]}; do
    for topic in ${TopicArray[@]}; do
        python3 scorer.py -g ${dir}/${topic}/scidr_dev.json -p ${dir}/${topic}/dev_out.json -v > ${dir}/${topic}/dev_res.txt
        python3 scorer.py -g ${dir}/${topic}/scidr_train.json -p ${dir}/${topic}/train_out.json -v > ${dir}/${topic}/train_res.txt
    done
done
