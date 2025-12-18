# CIFAR10
for s in  42 385 386 2024 2025
do
    for j in pxy  
    do
        for i in 0.2 0.3 0.4 0.5
        do
            python main.py \
                experiment=cifar10 \
                experiment.r=$i \
                desc="c10"$j"_final_"$i"_"$s \
                optim_goal=$j \
                seed=$s;
        done
    done
done

# CIFAR100
# for s in 42 385 386 2024 2025
# do
#     for j in pxy pyx
#     do
#         for i in 0.2 0.3 0.4 0.5
#         do
#             python main.py \
#                 experiment=cifar100 \
#                 experiment.r=$i \
#                 desc="c100"$j"_final_"$i"_"$s \
#                 optim_goal=$j \
#                 seed=$s;
#         done
#     done
# done

# CIFAR10N
# for j in pxy pyx
# do
#     for s in 42 385 386 2024 2025
#     do
#         for i in worse_label aggre_label random_label1 random_label2 random_label3
#         do
#             python main.py \
#                 experiment=cifar10n \
#                 target=$i \
#                 desc="c10N"$j"_wkl_"$i"_"$s \
#                 optim_goal=$j \
#                 seed=$s;
#         done
#     done
# done

# CIFAR100N
# for j in pxy pyx
# do
#     for s in 123 385 386 2024 2025
#     do
#         for i in noisy_label
#         do
#             python main.py \
#                 experiment=cifar100n \
#                 target=$i \
#                 desc="c100N"$j"_final_"$i"_"$s \
#                 optim_goal=$j \
#                 seed=$s;
#         done
#     done
# done

# RED
# for i in 0.2 0.4 0.6 0.8
# do
#     python main.py \
#         experiment=red \
#         experiment.r=$i \
#         desc="pyx_red_"$i \
#         root=/run/media/Data/red_blue/ \
#         optim_goal=pyx
# done

# ANIMAL
# for i in 2
# do
#     for j in pxy pyx
#     do
#         python main.py \
#             experiment=animal \
#             desc="A10"$j"_normgen" \
#             optim_goal=$j \
#             root=/media/hdd/fb/data_animal10n/ \
#             cot=$i;
#     done
# done
