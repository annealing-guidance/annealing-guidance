img2dataset \
  --url_list laion_top20k.csv \
  --input_format "csv" \
  --url_col "URL" \
  --caption_col "TEXT" \
  --output_format files \
  --output_folder ./laion_top20k_images \
  --processes_count 4 \
  --thread_count 16 \
  --resize_mode no
# 