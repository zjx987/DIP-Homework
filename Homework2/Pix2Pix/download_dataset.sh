FILE=$1

if [[ $FILE != "cityscapes" &&  $FILE != "night2day" &&  $FILE != "edges2handbags" && $FILE != "edges2shoes" && $FILE != "facades" && $FILE != "maps" ]]; then
  echo "Available datasets are cityscapes, night2day, edges2handbags, edges2shoes, facades, maps"
  exit 1
fi

echo "Specified [$FILE]"

URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE

#!/bin/bash

# 指定文件夹路径
FOLDER_PATH="./datasets/cityscapes/train"
# 指定输出的txt文件路径
OUTPUT_FILE="train_list_city.txt"

# 清空或创建输出文件
> "$OUTPUT_FILE"

# 遍历文件夹中的所有图片文件并按顺序写入
shopt -s nullglob  # 启用 nullglob，以便没有匹配时不会返回原始模式
for IMAGE in "$FOLDER_PATH"/*.{jpg,jpeg,png,gif}; do
    # 检查文件是否存在
    if [ -e "$IMAGE" ]; then
        echo "$IMAGE" >> "$OUTPUT_FILE"
    fi
done

# 对输出文件进行排序
sort -o "$OUTPUT_FILE" "$OUTPUT_FILE"

echo "所有图片路径已按顺序写入 $OUTPUT_FILE"
