CUDA_VISIBLE_DEVICES=2 python dataset_creation/generate_img_dataset.py --out_dir data/instruct-pix2pix-dataset-000 --prompts_file prompts/human-written-prompts.jsonl

CUDA_VISIBLE_DEVICES=0 python dataset_creation/generate_img_dataset.py --out_dir data/instruct-pix2pix-dataset-005 --prompts_file prompts/dialogs_all.jsonl