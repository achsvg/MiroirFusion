__kernel void test()
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
}