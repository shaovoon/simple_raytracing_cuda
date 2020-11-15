//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#ifndef MATERIALH
#define MATERIALH 

struct hit_record;

#include "ray.h"
#include "RandomNumGen.h"


float schlick(float cosine, float ref_idx) {
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else 
        return false;
}


vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2*dot(v,n)*n;
}


vec3 random_in_unit_sphere(const RandAccessor& rand) {
    vec3 p;
    do {
        p = 2.0*vec3(rand.Get(), rand.Get(), rand.Get()) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);
    return p;
}

class material;

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	const material* mat;
};

enum class material_type
{
    lambertian,
    metal,
    dielectric
};

class material  {
    public:
		material() = default;
		material(material_type mtype, const vec3& a, float f, float ri) : mat_type(mtype), albedo(a), ref_idx(ri) {
			if (f < 1) 
				fuzz = f; 
			else 
				fuzz = 1; 
		}
		bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, const RandAccessor& rand) const
		{
			switch (mat_type)
			{
			case material_type::lambertian:
				return lambertian_scatter(r_in, rec, attenuation, scattered, rand);
			case material_type::metal:
				return metal_scatter(r_in, rec, attenuation, scattered, rand);
			case material_type::dielectric:
				return dielectric_scatter(r_in, rec, attenuation, scattered, rand);
			}
		}

		bool lambertian_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, const RandAccessor& rand) const {
			vec3 target = rec.p + rec.normal + random_in_unit_sphere(rand);
			scattered = ray(rec.p, target - rec.p);
			attenuation = albedo;
			return true;
		}
		bool metal_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, const RandAccessor& rand) const {
			vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
			scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(rand));
			attenuation = albedo;
			return (dot(scattered.direction(), rec.normal) > 0);
		}
		bool dielectric_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, const RandAccessor& rand) const {
			vec3 outward_normal;
			vec3 reflected = reflect(r_in.direction(), rec.normal);
			float ni_over_nt;
			attenuation = vec3(1.0, 1.0, 1.0);
			vec3 refracted;
			float reflect_prob;
			float cosine;
			if (dot(r_in.direction(), rec.normal) > 0) {
				outward_normal = -rec.normal;
				ni_over_nt = ref_idx;
				//         cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
				cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
				cosine = sqrt(1 - ref_idx * ref_idx * (1 - cosine * cosine));
			}
			else {
				outward_normal = rec.normal;
				ni_over_nt = 1.0 / ref_idx;
				cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
			}
			if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
				reflect_prob = schlick(cosine, ref_idx);
			else
				reflect_prob = 1.0;
			if (rand.Get() < reflect_prob)
				scattered = ray(rec.p, reflected);
			else
				scattered = ray(rec.p, refracted);
			return true;
		}

        material_type mat_type;
		vec3 albedo;
		float fuzz;
		float ref_idx;

};

#endif




