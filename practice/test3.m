A = [2 3 4; 4 3 1; 4 7 4];
B = [2 4 6; 7 1 4; 9 0 3];
C = bitxor(A,B);
A = bitxor(C,B);
disp(A)
