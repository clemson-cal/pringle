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
#define VAPOR_USE_SHARED_PTR_ALLOCATOR // needed for OpenMP
#include <cmath>
#include <random>
#include "vapor/vapor.hpp"
constexpr auto SIGMA_FLOOR = 1e-12;

using namespace vapor;
using d_array_t = memory_backed_array_t<1, double, std::shared_ptr>;
// using d_array_t = memory_backed_array_t<1, double, ref_counted_ptr_t>;
using Product = d_array_t;
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))

// Simple array sum utility (host-side)
// Overload for summing realized device/host arrays
static double sum_array(const d_array_t& a)
{
    double s = 0.0;
    for (int i = 0; i < int(a.size()); ++i) {
        s += a[i];
    }
    return s;
}

// Template sum for mapped arrays not yet cached into d_array_t
template <typename Arr>
static double sum_array(const Arr& a)
{
    double s = 0.0;
    for (int i = 0; i < int(a.size()); ++i) {
        s += a[i];
    }
    return s;
}




/**
 * User configuration
 */
struct Config
{
    int fold = 1;
    int kernel_radius = 0; // zones included in the transport integral; zero means use the whole grid, N^2
    double initial_ring_mass = 1.0;
    double viscosity = 1.0 / 6.0;
    // double n = 0.0; // viscosity profile; n=1/2 for alpha
    double cpi = 1.0;
    double spi = 0.0;
    double tsi = 0.0;
    double cfl = 1.0;
    double tstart = 1.0;
    double tfinal = 1.0;
    vec_t<double, 3> domain = {0.1, 10.0, 0.1}; // inner, outer, step
    std::vector<uint> sp = {};
    std::vector<uint> ts = {};
    std::string outdir = ".";
};
VISITABLE_STRUCT(Config,
    fold,
    kernel_radius,
    initial_ring_mass,
    viscosity,
    // n,
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
    // CHANGE: track total mass injected so it can be plotted and compared
    double mass_injected;
    double mass_accreted;
    double angm_accreted;
    double mass_expelled;
    double angm_expelled;
    d_array_t mass;
    d_array_t angm;
};
VISITABLE_STRUCT(State,
    time,
    iter,
    mass_injected,
    mass_accreted,
    angm_accreted,
    mass_expelled,
    angm_expelled,
    mass,
    angm
);




//
// Evaluate the real-valued modified Bessel function using a naive series
//
static HD auto besselij(double j, double z, int num_terms=10) -> double
{
    double y = 0.0;
    for (int k = 0; k < num_terms; k++) {
        y += pow(0.5 * z, 2.0 * k) / (tgamma(k + 1.0) * tgamma(k + j + 1.0));
    }
    return y * pow(0.5 * z, j);
}




template <typename F>
static HD double integrate(F func, double a, double b)
{
    // Weights and abscissas for 4th-order Gaussian quadrature
    constexpr double w[4] = {+0.3478548451, +0.6521451549, +0.6521451549, +0.3478548451};
    constexpr double x[4] = {-0.8611363116, -0.3399810436, +0.3399810436, +0.8611363116};

    double y = 0.0;
    double m = 0.5 * (a + b);
    double h = 0.5 * (b - a);

    for (int i = 0; i < 4; ++i) {
        y += w[i] * func(m + h * x[i]);
    }
    return y * h;
}


template <typename A>
static auto diff(A y)
{
    return range(y.size() - 1).map([y] (int i) {
        return y[i + 1] - y[i];
    });
}



//
// Random helpers for per-timestep injection fluctuations
//
static auto rng() -> std::mt19937&
{
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

// Sample a truncated exponential on [amin, amax] with scale lambda (mean=lambda)
// Returned value is guaranteed within [amin, amax]
static double sample_truncated_exponential(double amin, double amax, double lambda)
{
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double u = U(rng());
    double umin = 1.0 - std::exp(-amin / lambda);
    double umax = 1.0 - std::exp(-amax / lambda);
    double ubar = u * (umax - umin) + umin;
    return -lambda * std::log(1.0 - ubar);
}

// Sample a symmetric, zero-mean fluctuation using a truncated exponential tail
// on [0, amax], and a random sign. The expected value is zero.
static double sample_symmetric_truncated_exponential(double amax, double lambda)
{
    double mag = sample_truncated_exponential(0.0, amax, lambda);
    std::uniform_int_distribution<int> B(0, 1);
    return B(rng()) ? mag : -mag;
}


//
// Equations for viscous spreading ring
//
// Arguments:
//   m   - total mass of the ring
//   r   - radius at which to evaluate the surface density
//   t   - time since the start of spreading
//   r0  - initial radius of the ring
//   nu  - kinematic viscosity
//
static HD double ring_sigma(double m, double r, double t, double r0, double nu)
{
    if (m == 0.0 || t == 0.0) {
        return SIGMA_FLOOR;
    }
    double tau = 12.0 * nu * t / (r0 * r0);
    double x = r / r0;
    double z = 2.0 * x / tau;
    double prefac = m / (M_PI * r0 * r0 * tau * pow(x, 0.25));
    if (z < 10.0) {
        return SIGMA_FLOOR + prefac * exp(-(1.0 + x * x) / tau) * besselij(0.25, z);
    } else {
        return SIGMA_FLOOR + prefac * exp(-(1.0 + x * x) / tau + z) / sqrt(2.0 * M_PI * z);
    }
}




static auto face_coordinates(const Config& config)
{
    auto r0 = config.domain[0];
    auto r1 = config.domain[1];
    auto dlogr = config.domain[2];
    auto ni = int(log(r1 / r0) / dlogr);
    auto ic = range(ni + 1);
    return ic.map([=] (int i) {
        return r0 * exp(dlogr * i);
    });
}

static auto initial_mass(const Config& config)
{
    auto t = config.tstart;
    auto m = config.initial_ring_mass;
    auto r0 = 1.0;
    auto nu = config.viscosity;
    auto sigma = [=] (double r) {
        return 2 * M_PI * r * ring_sigma(m, r, t, r0, nu);
    };
    auto rf = face_coordinates(config);
    return range(rf.size() - 1).map([rf, sigma] (int i) {
        auto r0 = rf[i];
        auto r1 = rf[i + 1];
        return integrate(sigma, r0, r1);
    }).cache();
}

static auto initial_angm(const Config& config)
{
    auto t = config.tstart;
    auto m = config.initial_ring_mass;
    auto r0 = 1.0;
    auto nu = config.viscosity;
    auto jdens = [=] (double r) {
        auto ell = sqrt(r);
        return 2 * M_PI * r * ring_sigma(m, r, t, r0, nu) * ell;
    };
    auto rf = face_coordinates(config);
    return range(rf.size() - 1).map([rf, jdens] (int i) {
        auto r0 = rf[i];
        auto r1 = rf[i + 1];
        return integrate(jdens, r0, r1);
    }).cache();
}




/**
 * Updates the state by evolving the mass and angular momentum arrays over
 * a time step dt.
 *
 * The mass array represents a sequence of N delta-function-like rings with
 * corresponding masses. Each ring expands over a time duration dt. The new
 * mass and angular momentum at time t + dt in each annulus are computed by
 * integrating the surface densities and angular momentum densities
 * contributed by all the spreading rings (from all other annuli) over the
 * annulus. The ring positions are obtained as the square of J / M in each
 * annulus.
 */
static void update_state(State& state, const Config& config)
{
    auto nu = config.viscosity;
    auto num_zones = int(state.mass.size());
    auto rf = face_coordinates(config).cache();
    auto M = state.mass;
    auto J = state.angm;
    auto kernel_radius = config.kernel_radius == 0 ? num_zones : config.kernel_radius;

    // NOTE (changed): injection location is currently fixed. Consider parameterizing.
    //auto injection_zone = num_zones / 10;
    auto injection_zone = 229;
    // CHANGE: baseline injection rate with zero-mean bounded random fluctuations
    auto base_mdot = 1.0;
    auto fluct = sample_symmetric_truncated_exponential(0.8, 0.3);
    auto mdot = base_mdot * (1.0 + fluct);

    auto rc = (J * J / M / M).cache();
    auto dt = config.cfl * (rc * rc / (12.0 * nu))[0];

    auto injected_mass = range(num_zones).map([=] (int i) {
        if (i == injection_zone) {
            return mdot * dt;
        }
        else {
            return 0.0;
        }
    });

    auto injected_angm = range(num_zones).map([=] (int i) {
        if (i == injection_zone) {
            return mdot * sqrt(rf[injection_zone]) * dt;
        }
        else {
            return 0.0;
        }
    });

    // CHANGE: fix accretion integral upper limit to rf[0] (inner boundary)
    auto mass_accreted = integrate([=] (double r) {
        auto s = 0.0;
        for (int j = 0; j < num_zones; ++j) {
            s += 2 * M_PI * r * ring_sigma(M[j], r, dt, rc[j], nu);
        }
        return s;
    }, 0.0, rf[0]);

    // Angular momentum flowing into the "void" inside the innermost zone
    auto angm_accreted = mass_accreted * sqrt(rf[0]);

    // Define the surface density function contributed by all rings
    auto new_sigma = [=] (double r, int i) {
        auto s = 0.0;
        auto j0 = max2(i - kernel_radius, 0);
        auto j1 = min2(i + kernel_radius, num_zones);
        for (int j = j0; j < j1; ++j) {
            s += ring_sigma(M[j], r, dt, rc[j], nu);
        }
        return s;
    };

    // CHANGE: outflow through the outer boundary estimated via conservation
    // New total mass before boundary accounting
    auto tentative_mass = range(num_zones).map([=] (int i) {
        return integrate([=] (double r) {
            return 2 * M_PI * r * new_sigma(r, i);
        }, rf[i], rf[i + 1]);
    });
    double total_mass_before = sum_array(M);
    double total_mass_after  = sum_array(tentative_mass.cache());
    double mass_loss_total   = total_mass_before - total_mass_after; // removed injected term
    double mass_expelled = max2(mass_loss_total - mass_accreted, 0.0);
    double angm_expelled = mass_expelled * sqrt(rf[num_zones]);

    // Compute the new mass in each annulus
    auto new_mass = range(num_zones).map([=] (int i) {
        return integrate([=] (double r) {
            return 2 * M_PI * r * new_sigma(r, i);
        }, rf[i], rf[i + 1]);
    }) + injected_mass;

    // Compute the new angular momentum in each annulus
    auto new_angm = range(num_zones).map([=] (int i) {
        return integrate([=] (double r) {
            auto ell = sqrt(r);
            return 2 * M_PI * r * new_sigma(r, i) * ell;
        }, rf[i], rf[i + 1]);
    }) + injected_angm;

    // Update the state with the new mass and angular momentum arrays
    state.mass = new_mass.cache();
    state.angm = new_angm.cache();
    // CHANGE: accumulate mass injected in the timeseries
    state.mass_injected += mdot * dt;
    state.mass_accreted += mass_accreted;
    state.angm_accreted += angm_accreted;
    state.mass_expelled += mass_expelled;
    state.angm_expelled += angm_expelled;
    state.time += dt;
    state.iter += 1.0;
}




/**
 *
 */
class Ruffle : public Simulation<Config, State, Product>
{
public:
    const char* name() const override
    {
        return "ruffle";
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
        return false;
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
        state.time = config.tstart;
        state.iter = 0.0;
        state.mass = initial_mass(config);
        state.angm = initial_angm(config);
        state.mass_injected = 0.0;
        state.mass_accreted = 0.0;
        state.angm_accreted = 0.0;
        state.mass_expelled = 0.0;
        state.angm_expelled = 0.0;
    }
    void update(State& state) const override
    {
        update_state(state, config);
    }
    bool should_continue(const State& state) const override
    {
        return state.time < config.tfinal;
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
            case 0: return "r";
            case 1: return "sigma";
        }
        return nullptr;
    }
    Product compute_product(const State& state, uint column) const override
    {
        auto num_zones = int(state.mass.size());
        auto rf = face_coordinates(config).cache();
        auto M = state.mass;
        auto J = state.angm;
        auto rc = J * J / M / M;
        auto sigma = M / diff(rf * rf) / M_PI;
        switch (column) {
            case 0: return rc.cache();
            case 1: return sigma.cache();
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
            case 1: return "mass_injected";   // CHANGE: new series
            case 2: return "mass_accreted";
            case 3: return "angm_accreted";
            case 4: return "mass_expelled";
            case 5: return "angm_expelled";
        }
        return nullptr;
    }
    double compute_timeseries_sample(const State& state, uint column) const override
    {
        switch (column) {
            case 0: return state.time;
            case 1: return state.mass_injected;  // CHANGE: new series
            case 2: return state.mass_accreted;
            case 3: return state.angm_accreted;
            case 4: return state.mass_expelled;
            case 5: return state.angm_expelled;
        }
        return 0.0;
    }
    vec_t<char, 256> status_message(const State& state, double secs_per_update) const override
    {
        return format("[%04d] t=%lf kzps=%.4lf",
            get_iteration(state),
            state.time,
            1e-3 * state.mass.size() / secs_per_update);
    }
};




int main(int argc, const char **argv)
{
    try {
        return Ruffle().run(argc, argv);
    }
    catch (const std::exception& e) {
        vapor::print("[error] ");
        vapor::print(e.what());
        vapor::print("\n");
    }
    return 0;
}  

