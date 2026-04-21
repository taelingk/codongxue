// svco_infer.cpp
// Build: C++14
// Compatible with: ONNX Runtime 1.10.x style C++ API
//
// Linux build example:
// aarch64-linux-gnu-g++ -O2 -std=c++14 \
//   -I/home/shm/onnxruntime-linux-aarch64-1.10.0/include \
//   /home/shm/svco_project/svco_infer.cpp \
//   -L/home/shm/onnxruntime-linux-aarch64-1.10.0/lib \
//   -Wl,-rpath,'$ORIGIN' \
//   -Wl,-rpath-link,/home/shm/onnxruntime-linux-aarch64-1.10.0/lib \
//   -static-libstdc++ -static-libgcc -lonnxruntime \
//   -o /home/shm/svco_project/svco_infer_aarch64
// ------------------------------------------------------------

#include <array>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/stat.h>

// ============================ path utils (C++14 replacement for filesystem) ============================
static inline bool path_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

static inline char path_sep() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

static inline std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    char sep = path_sep();
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + sep + b;
}

// ============================ small utils ============================
static inline std::string trim(std::string s) {
    auto notspace = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

static inline std::string tolower_str(std::string s) {
    for (size_t i = 0; i < s.size(); ++i) {
        s[i] = (char)std::tolower((unsigned char)s[i]);
    }
    return s;
}

// Robust CSV line parser with quotes
static std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                // lookahead for escaped quote
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                out.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
    }
    out.push_back(cur);
    return out;
}

static double to_double_safe(const std::string& s, bool& ok) {
    std::string t = trim(s);
    if (t.empty()) { ok = false; return 0.0; }
    try {
        size_t pos = 0;
        double v = std::stod(t, &pos);
        ok = (pos > 0);
        return v;
    } catch (...) {
        ok = false;
        return 0.0;
    }
}

// gender mapping fallback
static double parse_gender_as_double(const std::string& s) {
    bool ok = false;
    double v = to_double_safe(s, ok);
    if (ok) return v;

    std::string t = tolower_str(trim(s));
    if (t == "m" || t == "male" || t == "man" || t == "1") return 1.0;
    if (t == "f" || t == "female" || t == "woman" || t == "0") return 0.0;
    // unknown -> 0
    return 0.0;
}

static inline bool is_nan(double x) { return std::isnan(x); }

// ============================ NPY loader (minimal) ============================
// Supports: .npy v1.0/v2.0, little-endian float32/float64, C-order, shape (N,125,3)
struct NpyArray {
    std::vector<size_t> shape;
    std::vector<float> data_f32;  // stored as float32 in memory
};

static uint16_t read_u16_le(std::istream& is) {
    unsigned char b0, b1;
    is.read((char*)&b0, 1);
    is.read((char*)&b1, 1);
    return (uint16_t)(b0 | (b1 << 8));
}

static uint32_t read_u32_le(std::istream& is) {
    unsigned char b[4];
    is.read((char*)b, 4);
    return (uint32_t)(b[0] | (b[1]<<8) | (b[2]<<16) | (b[3]<<24));
}

static std::vector<size_t> parse_shape_from_header(const std::string& header) {
    size_t p1 = header.find('(');
    size_t p2 = header.find(')', p1);
    if (p1 == std::string::npos || p2 == std::string::npos || p2 <= p1) {
        throw std::runtime_error("NPY header parse error: shape not found");
    }

    std::string inside = header.substr(p1 + 1, p2 - p1 - 1);
    std::vector<size_t> shape;
    std::stringstream ss(inside);

    while (ss.good()) {
        std::string tok;
        std::getline(ss, tok, ',');
        tok = trim(tok);
        if (tok.empty()) continue;
        shape.push_back((size_t)std::stoull(tok));
    }
    return shape;
}

static std::string parse_descr_from_header(const std::string& header) {
    size_t p = header.find("'descr'");
    if (p == std::string::npos) p = header.find("\"descr\"");
    if (p == std::string::npos) throw std::runtime_error("NPY header parse error: descr not found");

    size_t q = header.find(':', p);
    if (q == std::string::npos) throw std::runtime_error("NPY header parse error: descr ':' not found");

    size_t s = header.find_first_of("'\"", q + 1);
    if (s == std::string::npos) throw std::runtime_error("NPY header parse error: descr quote not found");

    char quote = header[s];
    size_t e = header.find(quote, s + 1);
    if (e == std::string::npos) throw std::runtime_error("NPY header parse error: descr end quote not found");

    return header.substr(s + 1, e - (s + 1));
}

static bool parse_fortran_order_from_header(const std::string& header) {
    size_t p = header.find("fortran_order");
    if (p == std::string::npos) throw std::runtime_error("NPY header parse error: fortran_order not found");

    size_t q = header.find(':', p);
    if (q == std::string::npos) throw std::runtime_error("NPY header parse error: fortran_order ':' not found");

    size_t r = header.find_first_not_of(" \t", q + 1);
    if (r == std::string::npos) throw std::runtime_error("NPY header parse error: fortran_order value not found");

    if (header.compare(r, 4, "True") == 0 || header.compare(r, 4, "true") == 0) return true;
    return false;
}

static NpyArray load_npy_float_as_f32(const std::string& npy_path) {
    std::ifstream is(npy_path.c_str(), std::ios::binary);
    if (!is) throw std::runtime_error("Cannot open npy file: " + npy_path);

    char magic[6];
    is.read(magic, 6);
    if (!is || std::string(magic, magic + 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid npy magic header");
    }

    unsigned char major = 0, minor = 0;
    is.read((char*)&major, 1);
    is.read((char*)&minor, 1);

    size_t header_len = 0;
    if (major == 1) {
        header_len = read_u16_le(is);
    } else if (major == 2) {
        header_len = read_u32_le(is);
    } else {
        throw std::runtime_error("Unsupported npy version: " + std::to_string((int)major));
    }

    std::string header(header_len, '\0');
    if (header_len > 0) {
        is.read(&header[0], (std::streamsize)header_len);
    }
    if (!is) throw std::runtime_error("Failed to read npy header");

    std::string descr = parse_descr_from_header(header);
    bool fortran = parse_fortran_order_from_header(header);
    if (fortran) throw std::runtime_error("NPY fortran_order=True not supported in this minimal loader");

    std::vector<size_t> shape = parse_shape_from_header(header);
    if (shape.size() != 3) throw std::runtime_error("Expected npy shape rank=3 (N,125,3)");
    if (!(shape[1] == 125 && shape[2] == 3)) {
        std::ostringstream oss;
        oss << "Expected shape (N,125,3), got (" << shape[0] << "," << shape[1] << "," << shape[2] << ")";
        throw std::runtime_error(oss.str());
    }

    size_t total = shape[0] * shape[1] * shape[2];

    NpyArray arr;
    arr.shape = shape;
    arr.data_f32.resize(total);

    if (descr == "<f4" || descr == "|f4") {
        is.read(reinterpret_cast<char*>(arr.data_f32.data()), (std::streamsize)(total * sizeof(float)));
        if (!is) throw std::runtime_error("Failed to read npy float32 data");
    } else if (descr == "<f8" || descr == "|f8") {
        std::vector<double> tmp(total);
        is.read(reinterpret_cast<char*>(tmp.data()), (std::streamsize)(total * sizeof(double)));
        if (!is) throw std::runtime_error("Failed to read npy float64 data");
        for (size_t i = 0; i < total; ++i) arr.data_f32[i] = (float)tmp[i];
    } else {
        throw std::runtime_error("Unsupported npy descr: " + descr);
    }

    return arr;
}

// ============================ ONNX inference model ============================
class SVCOInferenceModel {
public:
    explicit SVCOInferenceModel(const std::string& model_dir)
        : model_dir_(model_dir),
          env_(ORT_LOGGING_LEVEL_WARNING, "svco_infer") {}

    void load_onnx_model() {
        std::string onnx_path = path_join(model_dir_, "resnet_se_lstm_model.onnx");
        if (!path_exists(onnx_path)) {
            throw std::runtime_error("ONNX model not found: " + onnx_path);
        }

        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring w(onnx_path.begin(), onnx_path.end());
        session_ = Ort::Session(env_, w.c_str(), so);
#else
        session_ = Ort::Session(env_, onnx_path.c_str(), so);
#endif

        // cache input/output names (ORT 1.10 old C++ API)
        Ort::AllocatorWithDefaultOptions allocator;

        // inputs
        size_t n_in = session_.GetInputCount();
        input_names_.clear();
        input_names_.reserve(n_in);
        for (size_t i = 0; i < n_in; ++i) {
            char* name_ptr = session_.GetInputName(i, allocator);
            if (!name_ptr) {
                throw std::runtime_error("GetInputName returned null");
            }
            input_names_.push_back(name_ptr);
            allocator.Free(name_ptr);
        }

        // outputs
        size_t n_out = session_.GetOutputCount();
        output_names_.clear();
        output_names_.reserve(n_out);
        for (size_t i = 0; i < n_out; ++i) {
            char* name_ptr = session_.GetOutputName(i, allocator);
            if (!name_ptr) {
                throw std::runtime_error("GetOutputName returned null");
            }
            output_names_.push_back(name_ptr);
            allocator.Free(name_ptr);
        }

        std::cout << "ONNX model loaded: " << onnx_path << "\n";
        std::cout << "Inputs (" << input_names_.size() << "): ";
        for (size_t i = 0; i < input_names_.size(); ++i) std::cout << input_names_[i] << " ";
        std::cout << "\nOutputs (" << output_names_.size() << "): ";
        for (size_t i = 0; i < output_names_.size(); ++i) std::cout << output_names_[i] << " ";
        std::cout << "\n";
    }

    // raw_signal: pointer to 125*3 float (already (125,3)).
    // clinical_features: [age, gender, weight, height, bsa, bmi, hr, sbp, dbp, pp]
    std::pair<float, float> infer_single(const float* raw_signal_125x3,
                                         const std::array<float, 10>& cf) {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // keep scalar buffers alive during Run
        scalars_.assign(cf.begin(), cf.end());

        std::vector<const char*> in_names_c;
        in_names_c.reserve(input_names_.size());

        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names_.size());

        for (size_t i = 0; i < input_names_.size(); ++i) {
            const std::string& nm = input_names_[i];
            in_names_c.push_back(nm.c_str());

            if (nm == "signal_input") {
                std::vector<int64_t> sig_shape(3);
                sig_shape[0] = 1;
                sig_shape[1] = 125;
                sig_shape[2] = 3;

                input_tensors.push_back(
                    Ort::Value::CreateTensor<float>(
                        mem,
                        const_cast<float*>(raw_signal_125x3),
                        125 * 3,
                        sig_shape.data(),
                        sig_shape.size()
                    )
                );
            } else {
                int scalar_idx = get_scalar_index_by_name(nm);
                if (scalar_idx < 0 || scalar_idx >= 10) {
                    std::ostringstream oss;
                    oss << "Model expects input name '" << nm
                        << "' which is not provided by this code.\n"
                        << "Please check ONNX input names above and adjust mapping.";
                    throw std::runtime_error(oss.str());
                }

                std::vector<int64_t> sh(2);
                sh[0] = 1;
                sh[1] = 1;

                input_tensors.push_back(
                    Ort::Value::CreateTensor<float>(
                        mem,
                        &scalars_[(size_t)scalar_idx],
                        1,
                        sh.data(),
                        sh.size()
                    )
                );
            }
        }

        std::vector<const char*> out_names_c;
        out_names_c.reserve(output_names_.size());
        for (size_t i = 0; i < output_names_.size(); ++i) {
            out_names_c.push_back(output_names_[i].c_str());
        }

        std::vector<Ort::Value> outputs = session_.Run(
            Ort::RunOptions{nullptr},
            in_names_c.data(),
            input_tensors.data(),
            input_tensors.size(),
            out_names_c.data(),
            out_names_c.size()
        );

        if (outputs.empty()) {
            throw std::runtime_error("No outputs returned by ONNX runtime.");
        }

        float pred_sv = 0.0f;
        {
            Ort::Value& out0 = outputs[0];
            if (!out0.IsTensor()) throw std::runtime_error("Output[0] is not a tensor");
            float* p = out0.GetTensorMutableData<float>();
            pred_sv = p[0];
        }

        float hr = cf[6];
        float pred_co = (pred_sv * hr) / 1000.0f;
        return std::make_pair(pred_sv, pred_co);
    }

private:
    int get_scalar_index_by_name(const std::string& nm) const {
        if (nm == "age_input")    return 0;
        if (nm == "gender_input") return 1;
        if (nm == "weight_input") return 2;
        if (nm == "height_input") return 3;
        if (nm == "bsa_input")    return 4;
        if (nm == "bmi_input")    return 5;
        if (nm == "hr_input")     return 6;
        if (nm == "sbp_input")    return 7;
        if (nm == "dbp_input")    return 8;
        if (nm == "pp_input")     return 9;
        return -1;
    }

private:
    std::string model_dir_;
    Ort::Env env_;
    Ort::Session session_{nullptr};

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<float> scalars_;
};

// ============================ CSV load/save ============================
struct CsvTable {
    std::vector<std::string> header;
    std::vector<std::vector<std::string> > rows;
    std::unordered_map<std::string, size_t> col_index_lc; // lowercase -> index
};

static CsvTable load_csv(const std::string& csv_path) {
    std::ifstream is(csv_path.c_str());
    if (!is) throw std::runtime_error("Cannot open CSV: " + csv_path);

    CsvTable t;
    std::string line;

    if (!std::getline(is, line)) throw std::runtime_error("CSV is empty: " + csv_path);
    t.header = parse_csv_line(line);
    for (size_t i = 0; i < t.header.size(); ++i) {
        t.col_index_lc[tolower_str(trim(t.header[i]))] = i;
    }

    while (std::getline(is, line)) {
        std::vector<std::string> cols = parse_csv_line(line);
        if (cols.size() < t.header.size()) cols.resize(t.header.size());
        t.rows.push_back(cols);
    }

    std::cout << "Loaded CSV: " << csv_path << " rows=" << t.rows.size() << "\n";
    return t;
}

static void save_csv(const std::string& out_path, const CsvTable& t) {
    std::ofstream os(out_path.c_str());
    if (!os) throw std::runtime_error("Cannot write CSV: " + out_path);

    struct CellWriter {
        static void write(std::ofstream& os, const std::string& s) {
            bool need_quote =
                (s.find(',') != std::string::npos) ||
                (s.find('"') != std::string::npos) ||
                (s.find('\n') != std::string::npos);

            if (!need_quote) {
                os << s;
                return;
            }

            os << '"';
            for (size_t i = 0; i < s.size(); ++i) {
                char c = s[i];
                if (c == '"') os << "\"\"";
                else os << c;
            }
            os << '"';
        }
    };

    // header
    for (size_t i = 0; i < t.header.size(); ++i) {
        if (i) os << ",";
        CellWriter::write(os, t.header[i]);
    }
    os << "\n";

    // rows
    for (size_t r = 0; r < t.rows.size(); ++r) {
        const std::vector<std::string>& row = t.rows[r];
        for (size_t i = 0; i < t.header.size(); ++i) {
            if (i) os << ",";
            std::string cell = (i < row.size() ? row[i] : "");
            CellWriter::write(os, cell);
        }
        os << "\n";
    }
}

static bool find_first_existing_col(const CsvTable& t,
                                    const std::vector<std::string>& candidates,
                                    size_t& idx_out) {
    for (size_t i = 0; i < candidates.size(); ++i) {
        std::string c = tolower_str(trim(candidates[i]));
        std::unordered_map<std::string, size_t>::const_iterator it = t.col_index_lc.find(c);
        if (it != t.col_index_lc.end()) {
            idx_out = it->second;
            return true;
        }
    }
    return false;
}

static std::string get_cell(const CsvTable& t, const std::vector<std::string>& row, size_t idx) {
    (void)t;
    if (idx >= row.size()) return "";
    return row[idx];
}

static double get_numeric_required(const CsvTable& t,
                                   const std::vector<std::string>& row,
                                   const std::vector<std::string>& candidates,
                                   const std::string& field_name_for_error) {
    size_t idx;
    if (!find_first_existing_col(t, candidates, idx)) {
        std::ostringstream oss;
        oss << "Missing required column for " << field_name_for_error << ". Candidates: ";
        for (size_t i = 0; i < candidates.size(); ++i) oss << candidates[i] << " ";
        throw std::runtime_error(oss.str());
    }

    bool ok = false;
    double v = to_double_safe(get_cell(t, row, idx), ok);
    if (!ok) {
        std::ostringstream oss;
        oss << "Column '" << t.header[idx] << "' cannot be parsed as number for "
            << field_name_for_error << ". Value='" << get_cell(t, row, idx) << "'";
        throw std::runtime_error(oss.str());
    }
    return v;
}

static double get_numeric_optional(const CsvTable& t,
                                   const std::vector<std::string>& row,
                                   const std::vector<std::string>& candidates,
                                   bool& exists_and_valid) {
    size_t idx;
    if (!find_first_existing_col(t, candidates, idx)) {
        exists_and_valid = false;
        return 0.0;
    }

    bool ok = false;
    double v = to_double_safe(get_cell(t, row, idx), ok);
    exists_and_valid = ok;
    return v;
}

// ============================ main pipeline ============================
int main(int argc, char** argv) {
    try {
        // Same defaults as your Python main
        std::string MODEL_DIR  = "/home/shm/下载/CO/cnap/resnet_SE_LSTM/20260208_111846";
        std::string TEST_CSV   = path_join(MODEL_DIR, "datasets/test_set.csv");
        std::string X_TEST_NPY = path_join(MODEL_DIR, "datasets/X_test.npy");

        // Allow override via argv (optional)
        // usage: svco_infer <MODEL_DIR> <TEST_CSV_PATH> <X_TEST_NPY>
        if (argc >= 2) MODEL_DIR  = std::string(argv[1]);
        if (argc >= 3) TEST_CSV   = std::string(argv[2]);
        if (argc >= 4) X_TEST_NPY = std::string(argv[3]);

        // load model
        SVCOInferenceModel infer(MODEL_DIR);
        infer.load_onnx_model();

        // load data
        CsvTable csv = load_csv(TEST_CSV);
        NpyArray x = load_npy_float_as_f32(X_TEST_NPY);

        size_t N = x.shape[0];
        if (csv.rows.size() != N) {
            std::ostringstream oss;
            oss << "Sample count mismatch: CSV rows=" << csv.rows.size()
                << " vs NPY N=" << N;
            throw std::runtime_error(oss.str());
        }

        std::cout << "Loaded NPY: " << X_TEST_NPY
                  << " shape=(" << x.shape[0] << "," << x.shape[1] << "," << x.shape[2] << ")\n";

        // add output columns
        struct ColAdder {
            static void add(CsvTable& csv, const std::string& name) {
                csv.col_index_lc[tolower_str(name)] = csv.header.size();
                csv.header.push_back(name);
                for (size_t i = 0; i < csv.rows.size(); ++i) csv.rows[i].push_back("");
            }
        };

        ColAdder::add(csv, "infer_sv");
        ColAdder::add(csv, "infer_co");
        ColAdder::add(csv, "sv_diff");
        ColAdder::add(csv, "co_diff");

        // locate pred_sv/pred_co columns (optional but used for diff if exists)
        bool has_pred_sv = false, has_pred_co = false, has_true_sv = false, has_true_co = false;
        size_t idx_pred_sv = 0, idx_pred_co = 0, idx_true_sv = 0, idx_true_co = 0;

        has_pred_sv = find_first_existing_col(csv, std::vector<std::string>{"pred_sv","Pred_SV","PRED_SV"}, idx_pred_sv);
        has_pred_co = find_first_existing_col(csv, std::vector<std::string>{"pred_co","Pred_CO","PRED_CO"}, idx_pred_co);
        has_true_sv = find_first_existing_col(csv, std::vector<std::string>{"true_sv","True_SV","TRUE_SV"}, idx_true_sv);
        has_true_co = find_first_existing_col(csv, std::vector<std::string>{"true_co","True_CO","TRUE_CO"}, idx_true_co);

        size_t idx_infer_sv = csv.col_index_lc["infer_sv"];
        size_t idx_infer_co = csv.col_index_lc["infer_co"];
        size_t idx_sv_diff  = csv.col_index_lc["sv_diff"];
        size_t idx_co_diff  = csv.col_index_lc["co_diff"];

        double sum_sv_diff = 0.0, sum_co_diff = 0.0;
        double max_sv_diff = 0.0, max_co_diff = 0.0;
        size_t diff_count_sv = 0, diff_count_co = 0;

        std::cout << "\n开始测试集推理...\n";

        for (size_t i = 0; i < N; ++i) {
            const std::vector<std::string>& row = csv.rows[i];

            // clinical features
            double age = get_numeric_required(csv, row, std::vector<std::string>{"age","Age","AGE"}, "age");

            // gender may be string -> parse
            std::string gender_raw;
            {
                size_t idx_g;
                if (!find_first_existing_col(csv, std::vector<std::string>{"gender","Gender","sex","Sex","SEX"}, idx_g)) {
                    throw std::runtime_error("Missing required column for gender");
                }
                gender_raw = get_cell(csv, row, idx_g);
            }

            double gender = parse_gender_as_double(gender_raw);
            double weight = get_numeric_required(csv, row, std::vector<std::string>{"weight","Weight","WEIGHT"}, "weight");
            double height = get_numeric_required(csv, row, std::vector<std::string>{"height","Height","HEIGHT"}, "height");
            double bsa    = get_numeric_required(csv, row, std::vector<std::string>{"bsa","BSA"}, "bsa");
            double bmi    = get_numeric_required(csv, row, std::vector<std::string>{"bmi","BMI"}, "bmi");
            double hr     = get_numeric_required(csv, row, std::vector<std::string>{"hr","HR"}, "hr");
            double sbp    = get_numeric_required(csv, row, std::vector<std::string>{"sbp","SBP","sbp_mmHg","SBP_mmHg"}, "sbp");
            double dbp    = get_numeric_required(csv, row, std::vector<std::string>{"dbp","DBP","dbp_mmHg","DBP_mmHg"}, "dbp");

            bool pp_ok = false;
            double pp = get_numeric_optional(csv, row, std::vector<std::string>{"pp","PP","pp_mmHg","PP_mmHg"}, pp_ok);
            if (!pp_ok) pp = sbp - dbp;

            std::array<float, 10> cf = {{
                (float)age, (float)gender, (float)weight, (float)height, (float)bsa,
                (float)bmi, (float)hr, (float)sbp, (float)dbp, (float)pp
            }};

            // signal pointer at sample i
            const float* sig = x.data_f32.data() + i * (125 * 3);

            std::pair<float, float> infer_result = infer.infer_single(sig, cf);
            float pred_sv = infer_result.first;
            float pred_co = infer_result.second;

            // write back
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << pred_sv;
                csv.rows[i][idx_infer_sv] = oss.str();
            }
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << pred_co;
                csv.rows[i][idx_infer_co] = oss.str();
            }

            // diffs vs pred_sv/pred_co if present
            if (has_pred_sv) {
                bool ok = false;
                double pv = to_double_safe(get_cell(csv, row, idx_pred_sv), ok);
                if (ok) {
                    double d = std::fabs((double)pred_sv - pv);
                    sum_sv_diff += d;
                    diff_count_sv++;
                    max_sv_diff = std::max(max_sv_diff, d);

                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << d;
                    csv.rows[i][idx_sv_diff] = oss.str();
                } else {
                    csv.rows[i][idx_sv_diff] = "";
                }
            } else {
                csv.rows[i][idx_sv_diff] = "";
            }

            if (has_pred_co) {
                bool ok = false;
                double pc = to_double_safe(get_cell(csv, row, idx_pred_co), ok);
                if (ok) {
                    double d = std::fabs((double)pred_co - pc);
                    sum_co_diff += d;
                    diff_count_co++;
                    max_co_diff = std::max(max_co_diff, d);

                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << d;
                    csv.rows[i][idx_co_diff] = oss.str();
                } else {
                    csv.rows[i][idx_co_diff] = "";
                }
            } else {
                csv.rows[i][idx_co_diff] = "";
            }

            if ((i + 1) % 50 == 0) {
                std::cout << "进度: " << (i + 1) << "/" << N << " 样本已推理\n";
            }
        }

        std::cout << "\n================ 推理结果对比 ================\n";
        if (diff_count_sv > 0) {
            std::cout << "训练时预测SV vs 推理模型SV：平均绝对差 = "
                      << (sum_sv_diff / (double)diff_count_sv)
                      << " mL，最大绝对差 = " << max_sv_diff << " mL\n";
        } else {
            std::cout << "未找到 pred_sv 列或 pred_sv 无法解析，无法计算 SV 差异。\n";
        }

        if (diff_count_co > 0) {
            std::cout << "训练时预测CO vs 推理模型CO：平均绝对差 = "
                      << (sum_co_diff / (double)diff_count_co)
                      << " L/min，最大绝对差 = " << max_co_diff << " L/min\n";
        } else {
            std::cout << "未找到 pred_co 列或 pred_co 无法解析，无法计算 CO 差异。\n";
        }

        std::string out_csv = path_join(MODEL_DIR, "test_set_infer_result.csv");
        save_csv(out_csv, csv);
        std::cout << "\n推理结果已保存至: " << out_csv << "\n";

        // print first 5 compare columns if exist
        std::cout << "\n前5个样本对比示例：\n";
        if (has_true_sv && has_pred_sv && has_true_co && has_pred_co) {
            std::cout << "true_sv,pred_sv,infer_sv,true_co,pred_co,infer_co\n";
            for (size_t i = 0; i < std::min<size_t>(5, csv.rows.size()); ++i) {
                std::vector<std::string>& r = csv.rows[i];
                std::cout << get_cell(csv, r, idx_true_sv) << ","
                          << get_cell(csv, r, idx_pred_sv) << ","
                          << get_cell(csv, r, idx_infer_sv) << ","
                          << get_cell(csv, r, idx_true_co) << ","
                          << get_cell(csv, r, idx_pred_co) << ","
                          << get_cell(csv, r, idx_infer_co) << "\n";
            }
        } else {
            std::cout << "CSV中缺少 true_sv/pred_sv/true_co/pred_co 某些列，已仍保存 infer_sv/infer_co。\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "推理过程出错: " << e.what() << "\n";
        return 1;
    }
}