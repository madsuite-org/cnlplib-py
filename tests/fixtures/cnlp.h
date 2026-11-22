/* cnlp.h: cnlp, the C ABI for nonlinear programs.
 *
 * Version 0.2 (2026-11-22; 0.1 was 2026-08-14).  This header is the
 * specification.  formats/ specifies the files that go with it; example/ is a
 * worked model and consumer; conformance/ holds reference files and checks.
 *
 * MODEL.  A library carries models, each under a symbol prefix that is its
 * name.  A model is instantiated into instances (positive ids) and evaluated
 * through sparse callbacks.  Per-model symbols are declared with X-macros
 * ("##" pastes the prefix: CNLP_DECLARE_MODEL(qp) declares qp_new, qp_obj, ...):
 *
 *     CNLP_DECLARE_MODEL(P)          required: core + instantiation
 *     CNLP_DECLARE_NEW_STR(P)        entry point is a string
 *     CNLP_DECLARE_BUILDER(P)        structured instantiation data
 *     CNLP_DECLARE_LAYOUT(P)         named blocks, parameters
 *     CNLP_DECLARE_SENSE(P)          objective sense (0.2)
 *     CNLP_DECLARE_WRITERS(P)        solution writers (0.2)
 *     CNLP_DECLARE_READERS(P)        the model's own input formats (0.2)
 *     CNLP_DECLARE_DEVICE(P)         GPU evaluation (0.2)
 *     CNLP_DECLARE_SINGLE(P)         binary32 evaluation (0.2)
 *     CNLP_DECLARE_FORMATS(P)        other number formats (0.2)
 *
 * SEMANTICS follow NLPModels.jl: indices are 1-based; sparse structures are
 * COO; the Hessian is the lower triangle of
 * obj_weight * grad^2 f + sum_i y_i * grad^2 c_i; buffers are caller-allocated.
 * Every integer (ids, sizes, indices, counts, lengths, statuses) is int64_t.
 *
 * STATUSES.  0 = success, except functions returning an id, count or byte
 * length (0 id = instantiation failure, -1 = index out of range).  1 = invalid
 * id, unknown index or wrong block kind; 3 = wrong length; 4 = unsupported
 * precision; other nonzero = model-defined failure.  No exception or longjmp
 * crosses the boundary.
 *
 * STRINGS: a function returns the needed byte length and copies what fits in
 * `cap` (call with NULL, 0 to size).  UTF-8, not NUL-terminated.
 *
 * TIERS are optional and implemented fully or not at all; a consumer probes
 * one witness symbol per tier, and a missing tier is not an error.
 *
 * SCOPE: names, sizes, offsets, dimensions, argument types, descriptions;
 * never a modelling package's expression structure.
 */
#ifndef CNLP_H
#define CNLP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Library level (optional; witness: cnlp_nmodels) ──────────────────────
 * The only fixed-name symbols.  Model k's name (k in 0..n-1) is its symbol
 * prefix.  cnlp_abi_version returns major * 1000 + minor (2 for 0.2); a
 * library without it implements 0.1. */
int64_t cnlp_nmodels(void);
int64_t cnlp_model_name(int64_t k, uint8_t* buf, int64_t cap);
int64_t cnlp_abi_version(void);

/* ── Core + instantiation (required) ──────────────────────────────────────
 * The entry point is P##_new, P##_new_str or the builder; P##_argtype says
 * which, as comma-separated `type|description` items:
 *     ""                     fixed, P##_new (argument ignored)
 *     "int|size"             P##_new(n)
 *     "string|case file"     P##_new_str(s)
 *     "int|N,Table{i::int w::f64}|bus"   the builder, one item per field
 * P##_nargs counts the items.  P##_meta fills x0, lvar, uvar (nvar) and
 * lcon, ucon (ncon); +-INFINITY allowed. */
#define CNLP_DECLARE_MODEL(P)                                                 \
    int64_t P##_nargs(void);                                                  \
    int64_t P##_argtype(uint8_t* buf, int64_t cap);                           \
    int64_t P##_new(int64_t n);                  /* -> id > 0; 0 = failure */ \
    int64_t P##_nvar(int64_t id);                                             \
    int64_t P##_ncon(int64_t id);                                             \
    int64_t P##_nnzj(int64_t id);                                             \
    int64_t P##_nnzh(int64_t id);                                             \
    int64_t P##_meta(int64_t id, double* x0, double* lvar, double* uvar,      \
                     double* lcon, double* ucon);                             \
    int64_t P##_obj (int64_t id, const double* x, double* out);  /* 1     */ \
    int64_t P##_grad(int64_t id, const double* x, double* g);    /* nvar  */ \
    int64_t P##_cons(int64_t id, const double* x, double* c);    /* ncon  */ \
    int64_t P##_jac_structure(int64_t id, int64_t* rows, int64_t* cols);      \
    int64_t P##_jac (int64_t id, const double* x, double* vals); /* nnzj  */ \
    int64_t P##_hess_structure(int64_t id, int64_t* rows, int64_t* cols);     \
    int64_t P##_hess(int64_t id, const double* x, const double* y,            \
                     double obj_weight, double* vals);           /* nnzh  */

/* String entry point (a case-file path, say). */
#define CNLP_DECLARE_NEW_STR(P)                                               \
    int64_t P##_new_str(const char* s);          /* -> id > 0; 0 = failure */

/* ── Structured instantiation (optional; witness: P##_schema) ────────────
 * P##_schema returns JSON {"fields":[{"name","kind":"scalar"|"array"|"table",
 * "type":"i64"|"f64"} or "columns" for tables]}.  Open a builder, set each
 * field by name (tables by column), then instantiate.  Kinds and types are
 * checked, never coerced; P##_data_ready returns 1 iff the data is complete
 * and consistent. */
#define CNLP_DECLARE_BUILDER(P)                                               \
    int64_t P##_schema(uint8_t* buf, int64_t cap);                            \
    int64_t P##_data_begin(void);                /* -> builder id > 0      */ \
    int64_t P##_set_scalar_i64(int64_t b, const char* field, int64_t v);      \
    int64_t P##_set_scalar_f64(int64_t b, const char* field, double v);       \
    int64_t P##_set_array_i64 (int64_t b, const char* field,                  \
                               const int64_t* v, int64_t len);                \
    int64_t P##_set_array_f64 (int64_t b, const char* field,                  \
                               const double* v, int64_t len);                 \
    int64_t P##_set_col_i64   (int64_t b, const char* table, const char* col, \
                               const int64_t* v, int64_t len);                \
    int64_t P##_set_col_f64   (int64_t b, const char* table, const char* col, \
                               const double* v, int64_t len);                 \
    int64_t P##_data_ready    (int64_t b);       /* 1 iff complete         */ \
    int64_t P##_new_from_data (int64_t b);       /* -> id > 0; 0 = failure */

/* ── Named blocks (optional; witness: P##_nblocks) ────────────────────────
 * Blocks the model named.  P##_block fills [kind, offset, length, ndims,
 * dims...]: kind 0 variable, 1 constraint, 2 parameter; offset 0-based into
 * the solution, constraint or parameter vector.  Per instance; unnamed
 * blocks occupy the vectors but are not listed.  Parameters are live
 * per-instance state: P##_set_value applies to later evaluations; 1 for a
 * non-parameter block, 3 for a wrong length. */
#define CNLP_DECLARE_LAYOUT(P)                                                \
    int64_t P##_nblocks(int64_t id);                                          \
    int64_t P##_block(int64_t id, int64_t k, int64_t* out);                   \
    int64_t P##_block_name(int64_t id, int64_t k, uint8_t* buf, int64_t cap); \
    int64_t P##_get_value(int64_t id, int64_t k, double* vals, int64_t len);  \
    int64_t P##_set_value(int64_t id, int64_t k,                              \
                          const double* vals, int64_t len);

/* ── Objective sense (0.2; optional; witness: P##_minimize) ───────────────
 * 1 minimize, 0 maximize, -1 invalid id.  Without it consumers assume
 * minimize, so every maximizing model must export it.  P##_obj is f(x) in the
 * model's sense, never negated. */
#define CNLP_DECLARE_SENSE(P)                                                 \
    int64_t P##_minimize(int64_t id);

/* ── Solution writers (0.2; optional; witness: P##_nwriters) ──────────────
 * Writer k writes the model's own format (a MATPOWER case, say) as
 *     int64_t P##_write_<name>(int64_t id, const char* path, const double* x,
 *         const double* y, const double* zl, const double* zu,
 *         double objective, const char* status);
 * host arrays of length nvar, ncon, nvar, nvar, multipliers in the sign
 * convention of formats/solution.md; status may be NULL.  P##_default_writer
 * names the default (0 = none: the standard cnlp-solution format). */
#define CNLP_DECLARE_WRITERS(P)                                               \
    int64_t P##_nwriters(void);                                               \
    int64_t P##_writer_name(int64_t k, uint8_t* buf, int64_t cap);            \
    int64_t P##_default_writer(int64_t id, uint8_t* buf, int64_t cap);

/* ── Readers (0.2; optional; witness: P##_nreaders) ───────────────────────
 * Reader k reads the model's own format; P##_reader_kind: 0 instance,
 * 1 parameters, 2 start, -1 out of range.  Reader <name> is exported as
 *     instance    int64_t P##_new_from_<name>(const char* path);   id or 0
 *     parameters  int64_t P##_set_from_<name>(int64_t id, const char* path);
 *     start       int64_t P##_start_from_<name>(int64_t id, const char* path,
 *                     double* x0, double* y0, double* zl, double* zu);
 * A start reader writes every entry (0 where the file has none).  Names are
 * unique across kinds.  P##_default_reader names the default instance reader
 * (0 = none); P##_new_str, if exported, is that reader. */
#define CNLP_DECLARE_READERS(P)                                               \
    int64_t P##_nreaders(void);                                               \
    int64_t P##_reader_name(int64_t k, uint8_t* buf, int64_t cap);            \
    int64_t P##_reader_kind(int64_t k);                                       \
    int64_t P##_default_reader(uint8_t* buf, int64_t cap);

/* ── Device evaluation (0.2; optional; witness: P##_device) ───────────────
 * P##_device(id): 1 = CUDA, device 0's primary context; -1 invalid id.
 * Arrays are device pointers, except P##_obj_d's `out` (one host double).
 * Structures and metadata stay on the host, in the order of the _d arrays.
 * Calls are synchronous. */
#define CNLP_DECLARE_DEVICE(P)                                                \
    int64_t P##_device(int64_t id);                                           \
    int64_t P##_obj_d (int64_t id, const double* x, double* out);            \
    int64_t P##_grad_d(int64_t id, const double* x, double* g);              \
    int64_t P##_cons_d(int64_t id, const double* x, double* c);              \
    int64_t P##_jac_d (int64_t id, const double* x, double* vals);           \
    int64_t P##_hess_d(int64_t id, const double* x, const double* y,         \
                       double obj_weight, double* vals);

/* ── binary32 (0.2; optional; witness: P##_sobj, and P##_sobj_d on a device)
 * The evaluation callbacks in C `float` (BLAS letters: unlettered = binary64,
 * s = binary32), on the same instances, sizes, structures and P##_meta. */
#define CNLP_DECLARE_SINGLE(P)                                                \
    int64_t P##_sobj (int64_t id, const float* x, float* out);   /* 1     */ \
    int64_t P##_sgrad(int64_t id, const float* x, float* g);     /* nvar  */ \
    int64_t P##_scons(int64_t id, const float* x, float* c);     /* ncon  */ \
    int64_t P##_sjac (int64_t id, const float* x, float* vals);  /* nnzj  */ \
    int64_t P##_shess(int64_t id, const float* x, const float* y,            \
                      float obj_weight, float* vals);            /* nnzh  */
#define CNLP_DECLARE_SINGLE_DEVICE(P)                                         \
    int64_t P##_sobj_d (int64_t id, const float* x, float* out);             \
    int64_t P##_sgrad_d(int64_t id, const float* x, float* g);               \
    int64_t P##_scons_d(int64_t id, const float* x, float* c);               \
    int64_t P##_sjac_d (int64_t id, const float* x, float* vals);            \
    int64_t P##_shess_d(int64_t id, const float* x, const float* y,          \
                        float obj_weight, float* vals);

/* ── Number formats (0.2; optional; witness: P##_nformats) ────────────────
 * Evaluation in the formats of formats/numbers.md.  P##_format_name(id, k)
 * gives format k's canonical name; arbitrary precision is listed as `apfloat`
 * and evaluated at `prec` bits (ignored, 0, for fixed formats).  The _x
 * callbacks take type-erased arrays of format-k elements with the core
 * lengths; `obj_weight` points to one element.  Status 1 for k out of range,
 * 4 for an unsupported prec. */
#define CNLP_DECLARE_FORMATS(P)                                               \
    int64_t P##_nformats(int64_t id);                                         \
    int64_t P##_format_name(int64_t id, int64_t k, uint8_t* buf, int64_t cap); \
    int64_t P##_xmeta(int64_t id, int64_t k, int64_t prec, void* x0,          \
                      void* lvar, void* uvar, void* lcon, void* ucon);         \
    int64_t P##_xobj (int64_t id, int64_t k, int64_t prec, const void* x,     \
                      void* out);                                             \
    int64_t P##_xgrad(int64_t id, int64_t k, int64_t prec, const void* x,     \
                      void* g);                                               \
    int64_t P##_xcons(int64_t id, int64_t k, int64_t prec, const void* x,     \
                      void* c);                                               \
    int64_t P##_xjac (int64_t id, int64_t k, int64_t prec, const void* x,     \
                      void* vals);                                            \
    int64_t P##_xhess(int64_t id, int64_t k, int64_t prec, const void* x,     \
                      const void* y, const void* obj_weight, void* vals);

/* DLPack {code, bits, lanes} of a format name (formats/numbers.md): 0 for
 * binary32/64/128 and their aliases, 1 otherwise. */
static inline int64_t cnlp_format_to_dlpack(const char* name, uint8_t* code,
                                            uint8_t* bits, uint16_t* lanes) {
    struct { const char* n; uint8_t b; } t[] = {
        {"binary32", 32}, {"f32", 32}, {"binary64", 64}, {"f64", 64},
        {"binary128", 128}, {"f128", 128}};
    for (unsigned i = 0; i < sizeof t / sizeof t[0]; i++) {
        const char* a = t[i].n; const char* b = name;
        while (*a && *a == *b) { a++; b++; }
        if (*a == 0 && *b == 0) { *code = 2; *bits = t[i].b; *lanes = 1; return 0; }
    }
    return 1;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CNLP_H */
