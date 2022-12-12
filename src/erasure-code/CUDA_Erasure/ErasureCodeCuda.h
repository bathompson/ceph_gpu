#ifndef CEPH_ERASURE_CODE_CUDA_H
#define CEPH_ERASURE_CODE_CUDA_H

#include "erasure-code/ErasureCode.h"

class ErasureCodeCuda : public ceph::ErasureCode
{
    public:
        int k;
        std::string DEFAULT_K;
        int m;
        std::string DEFAULT_M;
        //w = 8 is the only supported value
        const int w = 8;
        std::string DEFAULT_W;

        const char *technique;
        std::string rule_root;
        std::string rule_failure_domain;
        bool per_chunk_alignment;

        explicit ErasureCodeCuda(const char *_technique) : 
        k(0),
        DEFAULT_K("2"),
        m(0),
        DEFAULT_M("1"),
        DEFAULT_W("8"),
        technique(_technique),
        per_chunk_alignment(false)
        {}

        ~ErasureCodeCuda() override {}

        unsigned int get_chunk_count() const override
        {
            return k+m;
        }

        unsigned int get_data_chunk_count() const override
        {
            return k;
        }

        unsigned int get_chunk_size(unsigned int object_size) const override;

        int encode_chunks(const std::set<int> &want_to_encode, std::map<int, ceph::buffer::list> *decoded) override;
        
        int decode_chunks(const std::set<int> &want_to_read,
                const std::map<int, ceph::buffer::list> &chunks,
                std::map<int, ceph::buffer::list> *decoded) override;
        int init(ceph::ErasureCodeProfile &profile, std::ostream *ss) override;

        virtual void cuda_erasure_encode(uint8_t **data,
                                         uint8_t **coding,
                                         int blocksize) = 0;
        virtual int cuda_erasure_decode(size_t numErasures, int *erasures,
                                        uint8_t **data,
                                        uint8_t **coding,
                                        int blocksize) = 0;
        virtual unsigned get_alignment() const = 0;
        virtual void prepare() = 0;
        
    protected:
        virtual int parse(ceph::ErasureCodeProfile &profile, std::ostream *ss);

};

class ErasureCodeCudaVandermonde : public ErasureCodeCuda
{
    public:
        ErasureCodeCudaVandermonde() : ErasureCodeCuda("vand")
        {
            DEFAULT_K = "7";
            DEFAULT_M = "3";
            DEFAULT_W = "8";
        }

        void cuda_erasure_encode(uint8_t **data,
                                uint8_t **coding,
                                int blocksize) override;
        int cuda_erasure_decode(size_t numErasures, int *erasures,
                                uint8_t **data,
                                uint8_t **coding,
                                int blocksize) override;
        unsigned get_alignment() const override;
        void prepare() override;

        private:
            int parse(ceph::ErasureCodeProfile &profile, std::ostream *ss) override;

};

#endif