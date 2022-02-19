split=train

echo "We are dealing with ${split}"

# rm -rf dset_gen/generated/raw_${split}
# mkdir dset_gen/generated/raw_${split}
# rm -rf dset_gen/generated/merged/${split}_tmp
# mkdir dset_gen/generated/merged/${split}_tmp
rm -rf dset_gen/generated/merged/${split}
mkdir dset_gen/generated/merged/${split}

# PYTHONPATH=./ python dset_gen/generator.py --split $split
# PYTHONPATH=./ python dset_gen/preprocess.py --split $split
PYTHONPATH=./ python dset_gen/convert_to_h5.py --split $split

echo "Dataset size of ${split} is:"
du -sh dset_gen/generated/merged/${split}