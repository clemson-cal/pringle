/**
================================================================================
Copyright 2024, Jonathan Zrake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
================================================================================
*/

#include <cmath>
#include "vapor/vapor.hpp"




/**
 * 
 */
using namespace vapor;
using d_array_t = memory_backed_array_t<1, double, ref_counted_ptr_t>;
using Product = d_array_t;

#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))
#define GM 1.0

// HD static inline double plm_minmod(
//     double yl,
//     double yc,
//     double yr,
//     double plm_theta)
// {
//     double a = (yc - yl) * plm_theta;
//     double b = (yr - yl) * 0.5;
//     double c = (yr - yc) * plm_theta;
//     return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
// }




/**
 * 
 */
struct Config
{
    int fold = 50;
    int rk = 1;
    double viscosity = 0.001;
    double cpi = 1.0;
    double spi = 1.0;
    double tsi = 0.1;
    vec_t<double, 3> domain = {0.0, 10.0, 0.01};
    vec_t<double, 3> trange = {0.0, 0.0};
    std::vector<uint> sp = {0, 1};
    std::vector<uint> ts;
    std::string outdir = ".";
};
VISITABLE_STRUCT(Config, fold, rk, viscosity, cpi, spi, tsi, domain, trange, sp, ts, outdir);




/**
 * 
 */
struct State
{
    double time;
    double iter;
    d_array_t mass;
};
VISITABLE_STRUCT(State, time, iter, mass);




// static State average(const State& a, const State& b, double x)
// {
//     return x == 1.0 ? a : State{
//         (a.time * (1.0 - x) + b.time * x),
//         (a.iter * (1.0 - x) + b.iter * x),
//         (a.mass * (1.0 - x) + b.mass * x).cache()
//     };
// }




// def keplerian_omega_log_derivative(R):
//     """
//     Return the shear profile d(Omega) / d(log R)
//     """
//     return -1.5 * sqrt(GM / R**3)


// def keplerian_omega(R):
//     return sqrt(GM / R**3)


HD auto specific_angular_momentum(double R)
{
    return sqrt(GM * R);
}

HD auto specific_angular_momentum_derivative(double R)
{
    return 0.5 * sqrt(GM / R);
}

// def radial_velocity(M, R, viscosity, model, t):
//     """
//     Return the radial gas velocity at the internal zone interfaces

//     v = (d/dR(R g) + tau) / (sigma R l')

//     where tau is the external torque per unit length.
//     """
//     Rc = 0.5 * (R[1:] + R[:-1])
//     s = sigma(M, R)
//     n = viscosity(Rc)
//     A = keplerian_omega_log_derivative(Rc)
//     g = Rc * s * n * A
//     m = specific_angular_momentum_derivative(R[1:-1])

//     try:
//         tau = model.external_torque_per_unit_length(M, R, t)
//     except AttributeError:
//         tau = 0.0

//     return (diff(Rc * g) / diff(Rc) + tau) / (0.5 * (s[1:] + s[:-1]) * m * R[1:-1])




static auto face_coordinates(const Config& config)
{
    auto x0 = config.domain[0];
    auto x1 = config.domain[1];
    auto dx = config.domain[2];
    auto ni = int((x1 - x0) / dx);
    auto ic = range(ni + 1);
    auto xf = ic * dx + x0;
    return xf;
}

static auto cell_coordinates(const Config& config)
{
    auto x0 = config.domain[0];
    auto x1 = config.domain[1];
    auto dx = config.domain[2];
    auto ni = int((x1 - x0) / dx);
    auto ic = range(ni);
    auto xf = (ic + 0.5) * dx + x0;
    return xf;
}

static auto cell_lengths(const Config& config)
{
    auto xf = face_coordinates(config);
    auto dl = range(xf.size() - 1).map([xf] HD (int i) {
        auto x0 = xf[i];
        auto x1 = xf[i + 1];
        return x1 - x0;
    });
    return dl;
}

static auto cell_surface_areas(const Config& config)
{
    auto xf = face_coordinates(config);
    auto da = range(xf.size() - 1).map([xf] HD (int i) {
        auto x0 = xf[i];
        auto x1 = xf[i + 1];
        return M_PI * (x1 * x1 - x0 * x0);
    });
    return da;
}

static void update_state(State& state, const Config& config)
{
    auto rc = cell_coordinates(config);
    auto ni = rc.size();
    auto iv = range(ni + 1);
    auto ic = range(ni);
    auto interior_faces = iv.space().contract(1);
    auto interior_cells = ic.space().contract(1);
    auto dr = cell_lengths(config);
    auto dt = min(dr) * 0.1; // TODO
    auto dm = state.mass;
    auto da = cell_surface_areas(config);
    auto sigma = dm / da;

    auto fhat = iv[interior_faces].map([sigma] HD (int i)
    {
        // auto ul = u[i - 1];
        // auto ur = u[i];
        // auto pl = p[i - 1];
        // auto pr = p[i];
        // return riemann_hlle(pl, pr, ul, ur);
        return 0.0;
    }).cache();

    auto delta_dm = ic[interior_cells].map([fhat] HD (int i)
    {
        auto fm = fhat[i];
        auto fp = fhat[i + 1];
        return -(fp - fm);
    }) / dr * dt;

    state = State{
        state.time + dt,
        state.iter + 1.0,
        (dm.at(interior_cells) + delta_dm).cache(),
    };
}




/**
 * 
 */
class Blast : public Simulation<Config, State, Product>
{
public:
    const char* name() const override
    {
        return "pringle";
    }
    const char* author() const override
    {
        return "Jonathan Zrake (Clemson)";
    }
    const char* description() const override
    {
        return "Simulates the evolution of a cold 1d gas disk";
    }
    const char* output_directory() const override
    {
        return config.outdir.data();
    }
    double get_time(const State& state) const override
    {
        return state.time;
    }
    uint get_iteration(const State& state) const override
    {
        return round(state.iter);
    }
    void initial_state(State& state) const override
    {
        auto initial_sigma = [*this] HD (double r)
        {
            auto Mdot = -1.0;
            auto Jdot = -1.0;
            auto pi = M_PI;
            auto nu = config.viscosity;
            auto j = specific_angular_momentum(r);
            auto s = (Jdot - Mdot * j) / (3 * pi * nu * j);
            return s;
        };
        auto da = cell_surface_areas(config);
        auto sigma = cell_coordinates(config).map(initial_sigma);
        state.time = config.trange[0];
        state.iter = 0.0;
        state.mass = cache(sigma * da);
        if (any(state.mass < 0.0)) {
            throw std::runtime_error("initial data has negative surface density");
        }
    }
    void update(State& state) const override
    {
        update_state(state, config);
    }
    bool should_continue(const State& state) const override
    {
        return state.time < config.trange[0];
    }
    uint updates_per_batch() const override
    {
        return config.fold;
    }
    double checkpoint_interval() const override
    { 
        return config.cpi;
    }
    double product_interval() const override
    {
        return config.spi;
    }
    double timeseries_interval() const override
    {
        return config.tsi;
    }
    std::set<uint> get_product_cols() const override
    {
        return std::set(config.sp.begin(), config.sp.end());
    }
    const char* get_product_name(uint column) const override
    {
        switch (column) {
        case 0: return "sigma";
        case 1: return "radius";
        }
        return nullptr;
    }
    Product compute_product(const State& state, uint column) const override
    {
        switch (column) {
        case 0: return cache(state.mass / cell_surface_areas(config));
        case 1: return cache(cell_coordinates(config));
        }
        return {};
    }
    std::set<uint> get_timeseries_cols() const override
    {
        return std::set(config.ts.begin(), config.ts.end());
    }
    const char* get_timeseries_name(uint column) const override
    {
        switch (column) {
        case 0: return "time";
        }
        return nullptr;
    }
    double compute_timeseries_sample(const State& state, uint column) const override
    {
        return state.time;
    }
    vec_t<char, 256> status_message(const State& state, double secs_per_update) const override
    {
        return format("[%04d] t=%lf Mzps=%.2lf",
            get_iteration(state),
            get_time(state),
            1e-6 * state.mass.size() / secs_per_update);
    }
};




int main(int argc, const char **argv)
{
    try {
        return Blast().run(argc, argv);
    }
    catch (const std::exception& e) {
        vapor::print("[error] ");
        vapor::print(e.what());
        vapor::print("\n");
    }
    return 0;
}
