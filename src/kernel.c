void kernel simple_add(global const int* A,
                       global const int* B,
                       global int* C)
{
  int x = get_global_id(0);
  C[x] = A[x] + B[x];
}
