OBJ_NAME="cow"
TEXT_PROMPT="a beautiful cow statue"
CAM_DIST=0.8

STYLE_IMAGE="round_bird/round_bird_4.jpg"
IP_STRENGTH=1.0
CN_STRENGTH=1.0

python scripts/generate_texture.py \
    --input_dir "data/meshes/${OBJ_NAME}" \
    --output_dir "outputs" \
    --obj_file "${OBJ_NAME}_fr-z_up-y.obj" \
    --prompt "${TXT_PROMPT}" \
    --style_img "data/texture_images/${STYLE_IMAGE}" \
    --style_img_bg_color 255 255 255 \
    --ip_adapter_path "./ip_adapter" \
    --ip_adapter_strength $IP_STRENGTH \
    --ip_adapter_n_tokens 16 \
    --controlnet_cond "canny" \
    --controlnet_strength $CN_STRENGTH \
    --use_cc_edges True \
    --use_depth_edges True \
    --use_normal_edges True \
    --add_view_to_prompt \
    --ddim_steps 50 \
    --guidance_scale 10 \
    --new_strength 1 \
    --update_strength 0.4 \
    --view_threshold 0.1 \
    --blend 0 \
    --dist $CAM_DIST \
    --num_viewpoints 36 \
    --viewpoint_mode predefined \
    --use_principle \
    --update_steps 20 \
    --update_mode heuristic \
    --seed 42 \
    --post_process \
    --tex_resolution "1k" \
    --use_objaverse # assume the mesh is normalized with y-axis as up and z-axis as front
