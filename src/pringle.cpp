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

Notes:
================================================================================
Time counts down from tstart and the binary merges at t = 0.0

The code has units in which t_dec = 1 and r_dec = 1

If t_dec and r_dec are known, then cone can solve for nu:

The code uses constant-nu viscosity with nu = 1 / 6:
    a = t^(1 / 4)
    adot = a / (4 * t)
    a / adot = tvisc(a)
    4 * t = 2 a^2 / 3 nu; t = a = 1
    nu = 1 / 6 * r_dec^2 / t_dec
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




/**
 * User configuration
 */
struct Config
{
    int fold = 1;
    double sink_rate = 1e3;
    double viscosity = 1.0 / 6.0;
    double cpi = 0.0;
    double spi = 1.0;
    double tsi = 0.1;
    double cfl = 0.1;
    double tstart = 10.0;
    double tfinal = 1.0;
    vec_t<double, 3> domain = {0.0, 10.0, 0.01}; // inner, outer, step
    std::vector<uint> sp = {0, 1};
    std::vector<uint> ts = {0, 1, 2};
    std::string outdir = ".";
};
VISITABLE_STRUCT(Config,
    fold,
    sink_rate,
    viscosity,
    cpi,
    spi,
    tsi,
    cfl,
    tstart,
    tfinal,
    domain,
    sp,
    ts,
    outdir
);




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




/**
 * Return the shear profile d(Omega) / d(log R)
 */
auto keplerian_omega_log_derivative(double R)
{
    return -1.5 * sqrt(GM / R / R / R);
}

/**
 * Keplerian orbital frequency at radius R
 */
auto keplerian_omega(double R)
{
    return sqrt(GM / R / R / R);
}

/**
 * Specific angular momentum l at radius R
 */
auto specific_angular_momentum(double R)
{
    return sqrt(GM * R);
}

/**
 * Specific angular momentum derivative, dl/dR, at radius R
 */
auto specific_angular_momentum_derivative(double R)
{
    return 0.5 * sqrt(GM / R);
}




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
    auto rf = face_coordinates(config);
    auto dl = range(rf.size() - 1).map([rf] (int i) {
        auto r0 = rf[i];
        auto r1 = rf[i + 1];
        return r1 - r0;
    });
    return dl;
}

static auto cell_surface_areas(const Config& config)
{
    auto rf = face_coordinates(config);
    auto da = range(rf.size() - 1).map([rf] (int i) {
        auto r0 = rf[i];
        auto r1 = rf[i + 1];
        auto pi = M_PI;
        return pi * (r1 * r1 - r0 * r0);
    });
    return da;
}

static auto binary_separation(double time)
{
    return pow(time, 0.25);
}

static auto mass_source_term(const d_array_t& dm, double time, const Config& config)
{
    auto rc = cell_coordinates(config);
    auto nu = config.viscosity;
    auto f0 = config.sink_rate; // sink rate, relative to local viscous rate
    auto a = binary_separation(time);
    auto f = f0 * 1.5 * nu / a / a;
    if (f0 != 0.0) {
        return range(dm.space()).map([=] (int i) {
            return -dm[i] * f * exp(-pow(rc[i] / a, 16.0));
        }).cache();
    }
    else {
        return zeros<double>(dm.space()).cache();
    }
}

static auto dm_dot(const d_array_t& dm, double time, const Config& config)
{
    auto rf = face_coordinates(config);
    auto rc = cell_coordinates(config);
    auto da = cell_surface_areas(config);
    auto iv = range(rf.space());
    auto ic = range(rc.space());
    auto nu = config.viscosity;
    auto mdot_outer = -1.0;
    auto mdot_inner =  0.0;
    auto sigma = cache(dm / da);
    auto A = rc.map(keplerian_omega_log_derivative);
    auto g = ic.map([rc, sigma, A, nu] (int i) {
        return rc[i] * sigma[i] * nu * A[i];
    }).cache();
    auto fhat = iv.map([=] (int i)
    {
        if (i == rc.size()) {
            return mdot_outer;
        }
        if (i == 0) {
            return mdot_inner;
        }
        auto r = rf[i];
        auto pi = M_PI;
        auto lp = specific_angular_momentum_derivative(r);
        auto rm = rc[i - 1];
        auto rp = rc[i + 0];
        auto sm = sigma[i - 1];
        auto sp = sigma[i + 0];
        auto gm = g[i - 1];
        auto gp = g[i + 0];
        auto v_hat = ((rp * gp - rm * gm) / (rp - rm)) / (0.5 * (sm + sp) * r * lp);
        auto s_hat = 0.5 * (sm + sp);
        if (s_hat < 0.0) {
            throw std::runtime_error(format("found negative sigma %f at position %f", s_hat, r));
        }
        return 2.0 * pi * r * s_hat * v_hat;
    }).cache();
    return ic.map([fhat] (int i)
    {
        auto fm = fhat[i];
        auto fp = fhat[i + 1];
        return fm - fp;
    }) + mass_source_term(dm, time, config);
}

static auto next_dm(const d_array_t& dm, double time, const Config& config, double dt)
{
    return cache(dm + (dm_dot(dm, time, config)) * dt);
}

static auto next_dm_implicit(const d_array_t& x0, double time, const Config& config, double dt)
{
    auto f = [x0, time, &config, dt] (auto x) {
        return x - x0 - dm_dot(x, time, config) * dt;
    };
    auto eps = 1e-12;
    auto tol = config.tol;
    auto x1 = next_dm(x0, time, config, dt);
    auto x2 = cache(x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0) + eps));
    auto iter = 0;
    while (max(f(x2)) > tol) {
        auto x3 = cache(x2 - f(x2) * (x2 - x1) / (f(x2) - f(x1) + eps));
        x1 = x2;
        x2 = x3;
        iter += 1;
        if (iter > 100) {
            throw std::runtime_error("implicit update is not converging, try a larger tol");
        }
    }
    return x2;
}

static void update_state(State& state, const Config& config, double& timestep)
{
    auto nu = config.viscosity;    
    auto dr = cell_lengths(config);
    auto dt = timestep = config.cfl * (dr * dr / nu)[0];

    state = State{
        state.time - dt,
        state.iter + 1.0,
        next_dm(state.mass, state.time, config, dt),
    };
}




/**
 * 
 */
class Pringle : public Simulation<Config, State, Product>
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
    bool use_persistent_session() const override
    {
        return true;
    }
    double get_time(const State& state) const override
    {
        return config.tstart - state.time;
    }
    uint get_iteration(const State& state) const override
    {
        return round(state.iter);
    }
    void initial_state(State& state) const override
    {
        auto initial_sigma = [*this] (double r)
        {
            auto a = binary_separation(config.tstart);
            auto ell = 1.0;
            auto pi = M_PI;
            auto nu = config.viscosity;
            auto mdot = exp(-3. / 5. * ell / pow(a, 5. / 6.));
            if (r > a) {
                return mdot / (3 * pi * nu) * (1.0 - ell * sqrt(a / r));
            } else {
                return 1e-9;
            }
        };
        auto da = cell_surface_areas(config);
        auto sigma = cell_coordinates(config).map(initial_sigma);
        state.time = config.tstart;
        state.iter = 0.0;
        state.mass = cache(sigma * da);
        if (any(state.mass < 0.0)) {
            throw std::runtime_error("initial data has negative surface density");
        }
    }
    void update(State& state) const override
    {
        update_state(state, config, timestep);
    }
    bool should_continue(const State& state) const override
    {
        return state.time > config.tfinal;
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
        case 0: return "radius";
        case 1: return "mdot";
        case 2: return "sigma";
        }
        return nullptr;
    }
    Product compute_product(const State& state, uint column) const override
    {
        switch (column) {
        case 0: return cache(cell_coordinates(config));
        case 1: return cache(mass_source_term(state.mass, state.time, config));
        case 2: return cache(state.mass / cell_surface_areas(config));
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
        case 1: return "mdot";
        }
        return nullptr;
    }
    double compute_timeseries_sample(const State& state, uint column) const override
    {
        switch (column) {
        case 0: return state.time;
        case 1: return -sum(mass_source_term(state.mass, state.time, config));
        }
        return 0.0;
    }
    vec_t<char, 256> status_message(const State& state, double secs_per_update) const override
    {
        return format("[%04d] t=%lf dt=%.6e Mzps=%.2lf",
            get_iteration(state),
            state.time,
            timestep,
            1e-6 * state.mass.size() / secs_per_update);
    }
private:
    mutable double timestep;
};




int main(int argc, const char **argv)
{
    try {
        return Pringle().run(argc, argv);
    }
    catch (const std::exception& e) {
        vapor::print("[error] ");
        vapor::print(e.what());
        vapor::print("\n");
    }
    return 0;
}
