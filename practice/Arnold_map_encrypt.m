function C  = Arnold_map_encrypt(E)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
M = 512;
N = 512;
C = zeros(M,N,3);
for itr = 1:63
    for k = 1:3
        for i = 1:M
            for j = 1:N
                newi = mod((2*i + j),M)+1;
                newj = mod((i + j),N)+1;
                C(newi,newj,k) = E(i,j,k);
            end
        end
    end
    E = C;
end
end
