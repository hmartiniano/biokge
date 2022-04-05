#!/bin/bash
entities=evaluate/full*/entities.tsv
relations=evaluate/full*/relations.tsv
embeddings=../evaluate/$(tail -n 1 evaluate/full*log | awk '{print $4}' )
echo $relations
echo $entities
echo $embeddings
terms=(C0004352 C1510586)
grep ENSG ${entities} | awk '{print $2}' > head.list
#for i in "C0004352" "C0856975" "C1298684" "C1510586" "C1846135" "C4749945" ;
rm terms.txt
for i in ${terms[@]}
do
  echo $i
  echo $i >> terms.txt 
  rm -f tail.list
  grep $i  ${entities} | awk '{print $2}' > tail.list
  #rel=$(grep $i go_h_train.tsv | grep ENSG | awk '{print $2}' | sort | uniq)
  #echo $rel
  #grep $rel relations.tsv | awk  '{print $1}' > rel.list
  dglke_predict --rel_mfile ${relations} \
                --entity_mfile ${entities} \
                --model_path ${embeddings} \
                --format 'h_r_t' \
                --data_files head.list rel.list tail.list \
                --score_func logsigmoid \
                --topK $(grep -c ENSG ${entities}) \
                --exec_mode 'all' \
                --raw_data
  cp result.tsv ${i}_result.tsv
done



