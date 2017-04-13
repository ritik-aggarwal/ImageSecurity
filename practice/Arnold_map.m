function C  = Arnold_map_encrypt(C)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
M = 512;
N = 512;
for k = 1:3
    for i = 1:M
        for j = 1:N
            newi = mod((2*i + j),M)+1;
            newj = mod((i + j),N)+1;
            C(newi,newj,k) = C(i,j,k);
        end
    end
end

end

