#include "common/debug.h"
#include "ErasureCodeCuda.h"

extern "C"
{
    #include "Erasure/Erasure.h"
}

#define LARGEST_VECTOR_WORDSIZE 16

#define dout_context g_ceph_context
#define dout_subsys ceph_subsys_osd
#undef dout_prefix
#define dout_prefix _prefix(_dout)

using std::ostream;
using std::map;
using std::set;

using ceph::bufferlist;
using ceph::ErasureCodeProfile;

static ostream& _prefix(std::ostream *_dout)
{
    return *_dout << "ErasureCodeCuda: ";
}

int ErasureCodeCuda::init(ErasureCodeProfile &profile, ostream *ss)
{
    int err = 0;
    dout(10) << "technique=" << technique << dendl;
    profile["technique"] = technique;
    err |= parse(profile, ss);
    if(err)
        return err;
    prepare();
    return ErasureCode::init(profile, ss);
}

int ErasureCodeCuda::parse(ErasureCodeProfile &profile, ostream *ss)
{
    int err = ErasureCode::parse(profile, ss);
    err |= to_int("k", profile, &k, DEFAULT_K, ss);
    err |= to_int("m", profile, &k, DEFAULT_M, ss);
    err |= to_int("w", profile, &w, DEFAULT_W, ss);
    if (chunk_mapping.size() > 0 && (int)chunk_mapping.size() != k + m) {
    *ss << "mapping " << profile.find("mapping")->second
	<< " maps " << chunk_mapping.size() << " chunks instead of"
	<< " the expected " << k + m << " and will be ignored" << std::endl;
    chunk_mapping.clear();
    err = -EINVAL;
  }
  err |= sanity_check_k_m(k, m, ss);
  return err;
}

unsigned int ErasureCodeCuda::get_chunk_size(unsigned int object_size)
{
    unsigned alignment = get_alignment();
    if(per_chunk_alignment)
    {
        unsigned chunk_size = object_size / k;
        if(object_size % k)
            chunk_size++;
        dout(20) << "get_chunk_size: chunk_size " << chunk_size
        << " must be modulo "<<alignment << dendl;
        ceph_assert(alignment <= chunk_size);
        unsigned modulo = chunk_size % alignment;
        if(modulo)
        {
            dout(10) << "get_chunk_size: " << chunk_size
	       << " padded to " << chunk_size + alignment - modulo << dendl;
           chunk_size += alignment - modulo;
        }
        return chunk_size;
    }
    else
    {
        unsigned tail = object_size % alignment;
        unsigned padded_length = object_size + (tail ? (alignment - tail) : 0);
        ceph_assert(padded_length % k == 0);
        return padded_length / k;
    }
}

int ErasureCodeCuda::encode_chunks(const set<int> &want_to_encode,
                                    map<int, bufferlist> *encoded)
{
    char *chunks[k+m];
    for(int i = 0; i<k+m; i++)
    {
        chunks[i] = (*encoded)[i].c_str();
    }
    cuda_erasure_encode(&chunks[0], &chunks[k], (*encoded)[0].length());
}

int ErasureCodeCuda::decode_chunks(const set<int> &want_to_read,
				       const map<int, bufferlist> &chunks,
				       map<int, bufferlist> *decoded)
{
    unsigned blocksize = (*chunks.begin()).second.length();
    int erasures[k+m+1];
    int erasures_count = 0;
    char *data[k];
    char *coding[m];
    for(int i = 0; i<k+m; i++)
    {
        if(chunks.find(i) == chunks.end())
        {
            erasures[erasures_count] = i;
            erasures_count++;
        }
        if(i<k)
            data[i] = (*decoded)[i].c_str();
        else
            coding[i-k] = (*decoded)[i].c_str();
    }
    erasures[erasures_count] = -1;
    ceph_assert(erasures_count > 0);
    return cuda_erasure_decode(erasures_count, erasures, data, coding, blocksize);
}

ErasureCodeCudaVandermonde::cuda_erasure_encode(uint8_t **data, uint8_t **coding, int blocksize)
{
    encodeData(data, k, m, blocksize, coding);
}
ErasureCodeCudaVandermonde::cuda_erasure_decode(size_t numErasures, int *erasures, uint8_t **data, uint8_t **coding, int blocksize)
{
    decodeData(coding, erasures, )
}