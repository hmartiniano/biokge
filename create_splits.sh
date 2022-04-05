#/bin/bash
awk '{print $3}' gene_disease.tsv | sort | uniq -c | sort > sorted_disease_associations.txt 
awk '{if ($1 > 9) print $2}' sorted_disease_associations.txt > diseases_with_over_10_associations.txt
grep -f diseases_with_over_10_associations.txt gene_disease.tsv > gene_disease_over10.tsv 
shuf gene_disease_over10.tsv > gene_disease_over10_shuf.tsv
split -n l/3 gene_disease_over10_shuf.tsv 
cat goa.tsv goa_rna.tsv xaa | grep -v target | shuf > train.tsv
cat xab | shuf > test.tsv
cat xac | shuf > valid.tsv
cat gene_disease.tsv goa.tsv goa_rna.tsv | shuf > full.tsv
