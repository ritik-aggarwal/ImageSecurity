function C = LogisticMDS(P,para)

if max(P(:))>1
    S = 4;
else
    S = 32;
end


switch para
    case 'encryption'
        L = gf([4 2 1 3; 1 3 4 2; 2 4 3 1; 3 1 2 4],8);
    case 'decryption'
        L = gf([71 216 173 117; 173 117 71 216; 216 71 117 173; 117 173 216 71],8);
end
C = P;
mn = floor(size(P)/S)*S;
fun = @(y) double(y.x);
C(1:mn(1),1:mn(2)) = blkproc(P(1:mn(1),1:mn(2)),[4,4],@(y) feval(fun,L*y*L));

