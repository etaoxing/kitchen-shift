- NOTE: `postcorl_microwave_bottomknob_switch_slide/` contains duplicates of form `*'(1)'.mjl`
- NOTE: `postcorl_microwave_bottomknob_switch_slide/` and `friday_microwave_bottomknob_switch_slide/` use the same objects. No other demos use the same set of objects
- NOTE: `friday_topknob_bottomknob_hinge_slide/kitchen_playdata_2019_06_28_10_13_13.mjl` doesn't reach bottonknob properly, jitters, then resets to starting pos and demos the full task properly
- ERROR: cannot unpack `postcorl_microwave_bottomknob_switch_slide/kitchen_playdata_2019_07_11_13_34_20.mjl`, struct.error: unpack requires a buffer of 1728452 bytes

```
rm -Rf postcorl_microwave_bottomknob_switch_slide/*'(1)'*

rm -Rf friday_microwave_kettle_bottomknob_slide/postcorl_microwave_topknob_switch_hinge/kitchen_playdata_2019_07_11_11_48_15.mjl
rm -Rf friday_topknob_bottomknob_hinge_slide/postcorl_microwave_bottomknob_switch_slide/kitchen_playdata_2019_07_11_13_32_19.mjl
rm -Rf friday_topknob_bottomknob_hinge_slide/postcorl_microwave_bottomknob_switch_slide/kitchen_playdata_2019_07_11_13_34_20.mjl

# (demo actually manipulates switch, not hinge)
mv postcorl_kettle_topknob_bottomknob_hinge/kitchen_playdata_2019_07_11_18_02_01.mjl postcorl_kettle_topknob_bottomknob_switch/

# (doesn't move hinge)
rm friday_kettle_bottomknob_hinge_slide/kitchen_playdata_2019_06_28_13_46_02.mjl

# (doesn't move switch)
rm friday_microwave_bottomknob_switch_slide/kitchen_playdata_2019_06_28_12_39_25.mjl

# should give 581
find kitchen_demos_multitask/* -type f | sort > demofiles.txt
```
