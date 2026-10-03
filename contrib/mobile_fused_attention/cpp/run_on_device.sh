cd /data/local/tmp/mfa
mkdir -p $M.dir
# MODEL: e2b|e4b ; MAIN: fused tflite ; FUSED: --fused or empty ; CTX
ln -sf /data/local/tmp/mfa/${M}_Section2_TFLiteModel_tf_lite_embedder.tflite $M.dir/Section2_TFLiteModel_tf_lite_embedder.tflite
ln -sf /data/local/tmp/mfa/${M}_Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite $M.dir/Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite
for rep in ${REPS:-1 2}; do
LD_LIBRARY_PATH=/data/local/tmp/mfa ./mfa_engine $EXTRA --dir $M.dir --main $MAIN $FUSED --ctx ${CTX:-4096} --threads ${TH:-8} --ids p4k_ids.txt --max-new 64 --weight-cache $M.$TAG.wcache > $M.$TAG.$rep.log 2>&1 &
P=$!; f=$M.$TAG.$rep.mem; rm -f $f
while kill -0 $P 2>/dev/null; do dumpsys meminfo $P 2>/dev/null | grep -E '^ +(TOTAL PSS|TOTAL RSS)' | tr -s ' ' | tr '\n' '|' >> $f; echo >> $f; sleep 2; done
echo "== $M $TAG run $rep"; grep -E "^load |^prefill|^peak|^time|FATAL|rror" $M.$TAG.$rep.log | grep -v magic | head -6
echo "peak dumpsys PSS $(grep -o 'TOTAL PSS: [0-9]*' $f | awk '{print $3}' | sort -n | tail -1) KB, RSS $(grep -o 'TOTAL RSS: [0-9]*' $f | awk '{print $3}' | sort -n | tail -1) KB"
tail -1 $M.$TAG.$rep.log | cut -c1-120
done
