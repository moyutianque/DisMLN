split=train

rm -rf dset_gen/generated/raw_${split}/*
rm -rf dset_gen/generated/merged/${split}_tmp/*
rm -rf dset_gen/generated/merged/${split}/*

PYTHONPATH=./ python dset_gen/generator.py --split $split
PYTHONPATH=./ python dset_gen/preprocess.py --split $split
PYTHONPATH=./ python dset_gen/merge.py --split $split