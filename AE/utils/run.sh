declare -a LangArray=("french" "spanish" "persian" "danish" "vietnamese" "english")
declare -a TopicArray=("legal" "scientific")

for topic in ${TopicArray[@]}; do
    python3 code/baseline.py -input data/english/${topic}/dev.json -output output/english/${topic}/output_dev.json
    python3 code/baseline.py -input data/english/${topic}/train.json -output output/english/${topic}/output_train.json
    python3 scorer.py -g data/english/${topic}/dev.json -p output/english/${topic}/output_dev.json -v > result/english/${topic}/res_dev.txt
    python3 scorer.py -g data/english/${topic}/train.json -p output/english/${topic}/output_train.json -v > result/english/${topic}/res_train.txt
done

for language in ${LangArray[@]}; do
    python3 code/baseline.py -input data/${language}/dev.json -output output/${language}/output_dev.json
    python3 code/baseline.py -input data/${language}/train.json -output output/${language}/output_train.json
    python3 scorer.py -g data/${language}/dev.json -p output/${language}/output_dev.json -v > result/${language}/res_dev.txt
    python3 scorer.py -g data/${language}/train.json -p output/${language}/output_train.json -v > result/${language}/res_train.txt
done
