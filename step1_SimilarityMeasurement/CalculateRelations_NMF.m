load('CD.mat'); 

k = 50;
input_matrix = CD;

[circRNA_nmf, disease_nmf] = NMF(input_matrix, k);
save("circRNA_nmf.mat", "circRNA_nmf");
save("disease_nmf.mat", "disease_nmf");


function [circRNA_sim_nmf, disease_sim_nmf] = NMF(input_matrix, k)
    
    [W, H] = nnmf(double(input_matrix), k, 'algorithm', 'als', 'replicates', 5);
    
    circRNA_sim_nmf = 1 - pdist2(W, W, 'cosine');
    circRNA_sim_nmf = max(circRNA_sim_nmf, 0); 
    
    disease_sim_nmf = 1 - pdist2(H', H', 'cosine');
    disease_sim_nmf = max(disease_sim_nmf, 0);

end
