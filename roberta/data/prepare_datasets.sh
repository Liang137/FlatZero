wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar

echo "*** Use GLUE-SST-2 as default SST-2 ***"
mv original/SST-2 original/SST-2-original
mv original/GLUE-SST-2 original/SST-2

echo "*** Done with downloading datasets ***"

cd ..

for K in 32 256; do
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done