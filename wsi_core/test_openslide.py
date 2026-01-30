import openslide

wsi = openslide.open_slide('/media/local-admin/Elements/data_MAIA_2/CHC_CCA/HC_MAIA_boite5_CCA/13AG02766-25_MAIA01_HES.svs')

print(wsi.level_downsamples)