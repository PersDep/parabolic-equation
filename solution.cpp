#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

using namespace std;


const double M_PI_SQR = M_PI * M_PI;


double f0(double x, double y, double t, double xlim, double ylim);
double f1(double x, double y, double t, double xlim, double ylim);
double u0(double x, double y, double t, double xlim, double ylim);
double u1(double x, double y, double t, double xlim, double ylim);


class Data
{
    vector<vector<double> > data0;
    vector<vector<double> > data1;

    int grid_size;
    vector<int> halo;
    int local_size_x, local_size_y;
    int cur_proc_x, cur_proc_y;
    int proc_amount_x, proc_amount_y;
    int proc_rank, proc_amount;

    int neighbour(int shift_x, int shift_y);
    void sync_rows(int pos);
    void sync_columns(int pos);

public:
    Data(int grid_size, int proc_amount, int proc_rank,
         int proc_amount_x, int proc_amount_y);

    void Init(double xstep, double ystep, double xlim, double ylim);
    void Shift(vector<vector<double> > &&new_data0, vector<vector<double> > &&new_data1);
    void Sync();

    double operator()(int u, int i, int j) { if (u == 0) return data0[i][j]; return data1[i][j]; }
    int GetGlobalI(int i) { return cur_proc_y * grid_size / proc_amount_y + i; }
    int GetGlobalJ(int j) { return cur_proc_x * grid_size / proc_amount_x + j; }
    int GetGridSize() { return grid_size; }
    int GetLocalSizeX() { return local_size_x; }
    int GetLocalSizeY() { return local_size_y; }
    bool OnRoot() { return proc_rank == 0; }
};


class Solver
{
    Data data;

    double xlim, ylim;
    double xstep, ystep, eps;
    double xstepsqr, ystepsqr, xystep;
    int cur_iter;

    double laplas(int u, int i, int j);
    double nabla_div(int u, int i, int j);
    double inc(int u, int i, int j);

public:
    Solver(int grid_size, double xlim, double ylim,
           int proc_amount, int proc_rank,
           int proc_amount_x, int proc_amount_y);

    double Disrepancy();
    void Iterate();
};


Data::Data(int grid_size, int proc_amount, int proc_rank,
           int proc_amount_x, int proc_amount_y):
	grid_size(grid_size), proc_amount(proc_amount), proc_rank(proc_rank),
	proc_amount_x(proc_amount_x), proc_amount_y(proc_amount_y)
{
	cur_proc_y = proc_rank / proc_amount_x;
	cur_proc_x = proc_rank % proc_amount_x;
	local_size_x = grid_size / proc_amount_x + (cur_proc_x == proc_amount_x - 1) * (grid_size % proc_amount_x);
	local_size_y = grid_size / proc_amount_y + (cur_proc_y == proc_amount_y - 1) * (grid_size % proc_amount_y);

	for (int i = -1; i < 2; i++) halo.push_back(neighbour(i, 1));
	halo.push_back(neighbour(1, 0));
	for (int i = -1; i < 2; i++) halo.push_back(neighbour(-i, -1));
	halo.push_back(neighbour(-1, 0));
}

int Data::neighbour(int shift_x, int shift_y)
{
	return (cur_proc_x + shift_x + proc_amount_x) % proc_amount_x +
	       (cur_proc_y - shift_y + proc_amount_y) % proc_amount_y * proc_amount_x;
}

void Data::Init(double xstep, double ystep, double xlim, double ylim)
{
	data0.push_back(vector<double>(local_size_x + 2, 0));
	data1.push_back(vector<double>(local_size_x + 2, 0));
	for(int i = 1; i < local_size_y + 1; i++) {
		int global_i = GetGlobalI(i - 1);
		data0.push_back(vector<double>(1, 0)); data1.push_back(vector<double>(1, 0));
		for (int j = 1; j < local_size_x + 1; j++) {
			int global_j = GetGlobalJ(j - 1);
			data0[i].push_back(u0(global_j * xstep, global_i * ystep, 0, xlim, ylim));
			data1[i].push_back(u1(global_j * xstep, global_i * ystep, 0, xlim, ylim));
		}
		data0[i].push_back(0); data1[i].push_back(0);
	}
	data0.push_back(vector<double>(local_size_x + 2, 0));
	data1.push_back(vector<double>(local_size_x + 2, 0));
}

void Data::Shift(vector<vector<double> > &&new_data0, vector<vector<double> > &&new_data1)
{
	data0 = move(new_data0);
	data1 = move(new_data1);

	if (proc_amount == 1) {
		#pragma omp parallel for
		for(int i = 1; i < local_size_y + 1; i++)
			data1[1][i] = data1[local_size_y][i];
	} else {
		if (cur_proc_y == proc_amount_y - 1)
			MPI_Send(&data1[local_size_y][1], local_size_x,
			         MPI_DOUBLE, halo[5], 0, MPI_COMM_WORLD);
		if (cur_proc_y == 0)
			MPI_Recv(&data1[1][1], local_size_x,
			         MPI_DOUBLE, halo[1], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

void Data::sync_rows(int pos)
{
	int i_send = 1, i_recv = local_size_y + 1;
	if (pos == 5) i_send = local_size_y, i_recv = 0;
	MPI_Sendrecv(&data0[i_send][1], local_size_x, MPI_DOUBLE, halo[pos], 1,
	             &data0[i_recv][1], local_size_x, MPI_DOUBLE, halo[(pos + 4) % halo.size()],
	             1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	i_send = cur_proc_y ? 1 : 2;
	if (pos == 5) i_send = local_size_y - (cur_proc_y == proc_amount_y - 1 ? 1 : 0);
	MPI_Sendrecv(&data1[i_send][1], local_size_x, MPI_DOUBLE, halo[pos], 1,
	             &data1[i_recv][1], local_size_x, MPI_DOUBLE, halo[(pos + 4) % halo.size()],
	             1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void Data::sync_columns(int pos)
{
	vector<vector<double> > send(2);
	double recv[2][local_size_y];

	int j_send = local_size_x, j_recv = 0;
	if (pos == 7) j_send = 1, j_recv = local_size_x + 1;

	#pragma omp parallel for
	for (int i = 1; i < local_size_y + 1; i++) {
		send[0].push_back(data0[i][j_send]);
		send[1].push_back(data1[i][j_send]);
	}
	MPI_Sendrecv(send.data(), local_size_y * 2, MPI_DOUBLE, halo[pos], 1,
	             recv, local_size_y * 2, MPI_DOUBLE, halo[(pos + 4) % halo.size()],
	             1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	#pragma omp parallel for
	for (int i = 1; i < local_size_y + 1; i++) {
		data0[i][j_recv] = recv[0][i - 1];
		data1[i][j_recv] = recv[1][i - 1];
	}
}

void Data::Sync()
{
	sync_rows(1); sync_rows(5);
	sync_columns(3); sync_columns(7);

	vector<double> send(2), recv(2);
	for(size_t pos = 0; pos < halo.size(); pos += 2) {
		int i = 1, j = 1;
		if (pos == 2 || pos == 4) j *= local_size_x;
		if (pos == 6 || pos == 4) i *= local_size_y;
		send[0] = data0[i][j];
		send[1] = data1[i + (cur_proc_y == 0 && pos < 3) - (cur_proc_y == proc_amount_y - 1 && pos > 3)][j];
		MPI_Sendrecv(send.data(), 2, MPI_DOUBLE, halo[pos], 1,
		             recv.data(), 2, MPI_DOUBLE, halo[(pos + 4) % halo.size()],
		             1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		i = 0; j = 0;
		if (pos == 0 || pos == 2) i = local_size_y + 1;
		if (pos == 0 || pos == 6) j = local_size_x + 1;
		data0[i][j] = recv[0];
		data1[i][j] = recv[1];
	}
}

double f0(double x, double y, double t, double xlim, double ylim)
{
	return sin(M_PI * x / xlim) * sin(2 * M_PI * y / ylim) *
	       (cos(t) - M_PI_SQR * (1 / (xlim * xlim) + 4 / (ylim * ylim)) * sin(t));
}

double f1(double x, double y, double t, double xlim, double ylim)
{
	return cos(M_PI * x / xlim) * sin(2 * M_PI * y / ylim) *
	       (cos(t) - M_PI_SQR * (1 / (xlim * xlim) + 4 / (ylim * ylim)) * sin(t));
}

double u0(double x, double y, double t, double xlim, double ylim)
{
	return sin(M_PI * x / xlim) * sin(2 * M_PI * y / ylim) * sin(t);
}

double u1(double x, double y, double t, double xlim, double ylim)
{
	return cos(M_PI * x / xlim) * sin(2 * M_PI * y / ylim) * sin(t);
}

Solver::Solver(int grid_size, double xlim, double ylim,
               int proc_amount, int proc_rank,
               int proc_amount_x, int proc_amount_y) :
	data(grid_size, proc_amount, proc_rank, proc_amount_x, proc_amount_y), xlim(xlim), ylim(ylim), cur_iter(0)
{
	eps = numeric_limits<double>::min();
	xstep = xlim / (grid_size - 1);
	ystep = ylim / (grid_size - 1);
	xstepsqr = xstep * xstep;
	ystepsqr = ystep * ystep;
	xystep = 4 * xstep * ystep;
	data.Init(xstep, ystep, xlim, ylim);
	if (proc_rank == 0) {
		if (eps < min(xstepsqr, ystepsqr))
			cout << eps << " < " << min(xstepsqr, ystepsqr) << "; stability criterion passed" << endl;
		else
			cout << eps << " > " << min(xstepsqr, ystepsqr) << "; stability criterion failed" << endl;
	}
}

double Solver::Disrepancy()
{
	double diff = 0;
	for(int i = 1; i < data.GetLocalSizeY() + 1; i++) {
		int global_i = data.GetGlobalI(i - 1);
		for (int j = 1; j < data.GetLocalSizeX() + 1; j++) {
			int global_j = data.GetGlobalJ(j - 1);
			diff = max(abs(data(0, i, j) - u0(global_j * xstep, global_i * ystep,
			                                  cur_iter * eps, xlim, ylim)), diff);
			diff = max(abs(data(1, i, j) - u1(global_j * xstep, global_i * ystep,
			                                  cur_iter * eps, xlim, ylim)), diff);
		}
	}
	double global;
	MPI_Reduce(&diff, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (data.OnRoot()) cout << cur_iter << " iterations disrepancy: " << global << endl;
	return global;
}

double Solver::laplas(int u, int i, int j) { return (data(u, i, j - 1) - 2 * data(u, i, j) +
                                                     data(u, i, j + 1)) / (xstepsqr); }

double Solver::nabla_div(int u, int i, int j) { return (data(u, i - 1, j) - 2 * data(u, i, j) +
                                                        data(u, i + 1, j)) / (ystepsqr); }

double Solver::inc(int u, int i, int j) { return (data(u, i + 1, j + 1) - data(u, i - 1, j + 1) -
                                                  data(u, i + 1, j - 1) + data(u, i - 1, j - 1)) / (xystep); }

void Solver::Iterate()
{
	double tau = ++cur_iter * eps;
	data.Sync();
	vector<vector<double> > new_data0(1, vector<double>(data.GetLocalSizeX() + 2, 0));
	vector<vector<double> > new_data1(1, vector<double>(data.GetLocalSizeX() + 2, 0));
	#pragma omp parallel for
	for(int i = 1; i < data.GetLocalSizeY() + 1; i++) {
		int global_i = data.GetGlobalI(i - 1);
		new_data0.push_back(vector<double>(1, 0)); new_data1.push_back(vector<double>(1, 0));
		for(int j = 1; j < data.GetLocalSizeX() + 1; j++) {
			int global_j = data.GetGlobalJ(j - 1);
			if (global_j == 0) {
				new_data0[i].push_back(0);
				new_data1[i].push_back((4 * data(1, i, j + 1) - data(1, i, j + 2)) / 3);
			} else if (global_j == data.GetGridSize() - 1) {
				new_data0[i].push_back(0);
				new_data1[i].push_back((4 * data(1, i, j - 1) - data(1, i, j - 2)) / 3);
			} else {
				new_data0[i].push_back((laplas(0, i, j) + 2 * nabla_div(0, i, j) + inc(1, i, j) +
				                        f0(global_j * xstep, global_i * ystep, tau, xlim, ylim)) * eps
				                       + data(0, i, j));
				new_data1[i].push_back((laplas(1, i, j) + 2 * nabla_div(1, i, j) + inc(0, i, j) +
				                        f1(global_j * xstep, global_i * ystep, tau, xlim, ylim)) * eps
				                       + data(1, i, j));
			}
		}
		new_data0[i].push_back(0); new_data1[i].push_back(0);
	}
	new_data0.push_back(vector<double>(data.GetLocalSizeX() + 2, 0));
	new_data1.push_back(vector<double>(data.GetLocalSizeX() + 2, 0));
	data.Shift(move(new_data0), move(new_data1));
}


int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int proc_amount = 0, proc_rank = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &proc_amount);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	int grid_size = int(strtol(argv[1], nullptr, 10));
	int iterations = int(strtol(argv[2], nullptr, 10));
	int proc_x = int(strtol(argv[3], nullptr, 10));
	int proc_y = int(strtol(argv[4], nullptr, 10));

	int threads = int(strtol(argv[5], nullptr, 10));
	omp_set_num_threads(threads);

	Solver solution(grid_size, 1, 1, proc_amount, proc_rank, proc_x, proc_y);
	double time = MPI_Wtime();
	for (int i = 0; i < iterations; i++) solution.Iterate();
	time = MPI_Wtime() - time;
	solution.Disrepancy();

	if (proc_rank == 0) {
		cout << "Grid size: " << grid_size << endl
		     << "MPI processes: " << proc_amount << endl
		     << "OMP Threads: " << threads << endl
		     << "Time: " << time << endl;
	}

	MPI_Finalize();
	return 0;
}
