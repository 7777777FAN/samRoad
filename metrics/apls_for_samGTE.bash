# declare -a arr=(8 9 19 28 29 39 48 49 59 68 69 79 88 89 99 108 109 119 128 129 139 148 149 159 168 169 179)
declare -a arr=(49 179)

# source directory
dir=./save/修正损失计算重训/decode_result

echo "预测结果保存路径：${dir}"
mkdir -p ../$dir/results/apls
# now loop through the above array
for i in "${arr[@]}"   

do
    if test -f "../${dir}/region_${i}_graph.p"; then
        echo "========================$i======================"
        python ./apls/convert.py "../cityscale/20cities/region_${i}_graph_gt.pickle" gt.json
        python ./apls/convert.py "../${dir}/region_${i}_graph.p" prop.json
        go run ./apls/main.go gt.json prop.json ../$dir/results/apls/$i.txt 
    fi
done

python3 apls.py --dir $dir