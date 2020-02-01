void kernel crc_iter(global const int *iter_array,
                     global int *results)
{
  unsigned g = get_global_id(0);
  unsigned l = get_local_id(0);
  int idx = ( g == 0 ? 0 : ( l == 0 ? 1 : 2 ) );
  int iters = iter_array[idx];
  int i;
  unsigned x = g;
  unsigned poly = 0xEDB88320;

  for(i=0; i<iters; i++) {
    x = ( (x>>1) ^ ( (x&1) ? poly : 0 ) );
  }
  results[g] = x;
}
