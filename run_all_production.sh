

# python newer_parse.py /data/dockop_data/AmpC_screen_table_clean.feather /data/newdockop/dockop/code/mod_code_base/data_out/test

for fp in morgan; do
    for size in 4096; do
	python /data/newdockop/dockop/code/mod_code_base/main_production.py $fp $size /data/newdockop/dockop/code/mod_code_base/logreg_only.json
    done
done


# python newer_parse.py /data/dopamine_3_results/150M_fps
