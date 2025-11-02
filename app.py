"""
Streamlit App untuk ST-DBSCAN & K-Means Earthquake Clustering
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import radians, sin, cos, sqrt, atan2

from stdbscan import (STDBSCAN, calculate_clustering_metrics, analyze_clusters, 
                      generate_sample_data, find_optimal_eps1, plot_knn_distance)
from kmeans import (SpatioTemporalKMeans, find_optimal_k, calculate_kmeans_metrics, 
                    analyze_kmeans_clusters, analyze_kmeans_clusters_with_patterns)


def create_spatial_plot(df, labels):
    df_plot = df.copy()
    df_plot['cluster'] = labels
    df_plot['cluster_label'] = df_plot['cluster'].apply(lambda x: f'Cluster {x}' if x >= 0 else 'Noise')
    
    fig = px.scatter_mapbox(df_plot, lat='latitude', lon='longitude', color='cluster_label',
                            size='magnitude', hover_data=['datetime', 'magnitude', 'depth'],
                            color_discrete_sequence=px.colors.qualitative.Set1, zoom=4, height=500)
    fig.update_layout(mapbox_style="open-street-map", title="Distribusi Spasial Cluster")
    return fig


def create_temporal_plot(df, labels):
    df_plot = df.copy()
    df_plot['cluster'] = labels
    df_plot['cluster_label'] = df_plot['cluster'].apply(lambda x: f'Cluster {x}' if x >= 0 else 'Noise')
    
    fig = px.scatter(df_plot, x='datetime', y='latitude', color='cluster_label', size='magnitude',
                     hover_data=['longitude', 'magnitude', 'depth'],
                     color_discrete_sequence=px.colors.qualitative.Set1, height=400)
    fig.update_layout(title="Distribusi Temporal vs Latitude", xaxis_title="Waktu", yaxis_title="Latitude")
    return fig


def create_3d_plot(df, labels):
    df_plot = df.copy()
    df_plot['cluster'] = labels
    df_plot['cluster_label'] = df_plot['cluster'].apply(lambda x: f'Cluster {x}' if x >= 0 else 'Noise')
    df_plot['days'] = (df_plot['datetime'] - df_plot['datetime'].min()).dt.total_seconds() / 86400
    
    fig = px.scatter_3d(df_plot, x='longitude', y='latitude', z='days', color='cluster_label',
                        size='magnitude', hover_data=['datetime', 'magnitude', 'depth'],
                        color_discrete_sequence=px.colors.qualitative.Set1, height=600)
    fig.update_layout(title="Visualisasi 3D Spasial-Temporal",
                      scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Waktu (hari)"))
    return fig


def create_elbow_plot(optimal_k_results):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Method", "Silhouette Score"))
    
    k_values = optimal_k_results['k_values']
    fig.add_trace(go.Scatter(x=k_values, y=optimal_k_results['inertias'], mode='lines+markers',
                             marker=dict(size=10, color='blue'), name='Inertia'), row=1, col=1)
    fig.add_trace(go.Scatter(x=k_values, y=optimal_k_results['silhouettes'], mode='lines+markers',
                             marker=dict(size=10, color='green'), name='Silhouette'), row=1, col=2)
    
    fig.update_xaxes(title_text="K", row=1, col=1)
    fig.update_xaxes(title_text="K", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_comparison_chart(df, labels):
    cluster_stats = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        cluster_data = df[labels == cid]
        duration = (cluster_data['datetime'].max() - cluster_data['datetime'].min()).days
        cluster_stats.append({
            'cluster': f'C{cid}',
            'n_events': len(cluster_data),
            'duration': duration,
            'avg_mag': cluster_data['magnitude'].mean()
        })
    
    stats_df = pd.DataFrame(cluster_stats)
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Events", "Durasi (hari)", "Avg Magnitude"))
    colors = px.colors.qualitative.Set1
    
    fig.add_trace(go.Bar(x=stats_df['cluster'], y=stats_df['n_events'], 
                         marker_color=colors[:len(stats_df)], text=stats_df['n_events']), row=1, col=1)
    fig.add_trace(go.Bar(x=stats_df['cluster'], y=stats_df['duration'],
                         marker_color=colors[:len(stats_df)], text=stats_df['duration']), row=1, col=2)
    fig.add_trace(go.Bar(x=stats_df['cluster'], y=stats_df['avg_mag'],
                         marker_color=colors[:len(stats_df)], text=[f"{x:.2f}" for x in stats_df['avg_mag']]), row=1, col=3)
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def main():
    st.set_page_config(page_title="Earthquake Clustering", layout="wide", page_icon="🌍")
    
    st.title("🌍 Klasterisasi Spasial-Temporal Gempabumi")
    st.markdown("### ST-DBSCAN & K-Means Clustering")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # Data Input
        st.subheader("📁 Data")
        data_source = st.radio("Sumber data:", ["Upload CSV", "Data Contoh"])
        
        df = None
        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    st.session_state['df_input'] = df
                    st.success(f"✅ {len(df)} events")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            if st.button("🎲 Generate"):
                df = generate_sample_data()
                st.session_state['df_input'] = df
                st.success(f"✅ {len(df)} events")
        
        if 'df_input' in st.session_state:
            df = st.session_state['df_input']
        
        st.markdown("---")
        
        # Algorithm Selection
        st.subheader("🎯 Algoritma")
        algorithm = st.radio("Pilih:", ["ST-DBSCAN", "K-Means"])
        
        # Mode Selection
        clustering_mode = st.radio("Mode:", ["3D (Spasial-Temporal)", "4D (+Depth)"])
        use_depth = "4D" in clustering_mode
        
        st.markdown("---")
        
        # Parameters
        if algorithm == "ST-DBSCAN":
            st.subheader("⚙️ Parameter DBSCAN")
            
            # Auto-detect eps1 - INI YANG PENTING!
            auto_eps1 = st.checkbox("🔍 Auto eps1", value=False, 
                                    help="Otomatis mencari eps1 optimal menggunakan k-distance plot")
            
            if auto_eps1 and df is not None:
                with st.expander("⚙️ Konfigurasi Auto eps1"):
                    k_values = st.multiselect(
                        "Nilai k untuk analisis:",
                        [3, 4, 5, 6, 7, 8],
                        default=[4, 5, 6],
                        help="k biasanya = min_samples - 1"
                    )
                    if not k_values:
                        k_values = [4, 5, 6]
                    
                    if st.button("🔍 Analisis eps1 Optimal"):
                        with st.spinner("Menganalisis k-distance..."):
                            optimal_eps1_results = find_optimal_eps1(df, k_range=sorted(k_values))
                            st.session_state['optimal_eps1_results'] = optimal_eps1_results
                
                # Jika sudah ada hasil analisis
                if 'optimal_eps1_results' in st.session_state:
                    opt_results = st.session_state['optimal_eps1_results']
                    rec_eps1 = opt_results['recommendation']
                    
                    st.success(f"💡 Rekomendasi: {rec_eps1:.2f} km")
                    
                    eps1 = st.number_input(
                        "eps1 - Spasial (km):",
                        min_value=1.0,
                        max_value=500.0,
                        value=float(rec_eps1),
                        step=1.0,
                        help="Anda bisa override nilai rekomendasi"
                    )
                else:
                    st.info("Klik tombol 'Analisis eps1 Optimal' di atas")
                    eps1 = 50.0
            else:
                if auto_eps1 and df is None:
                    st.warning("Muat data dulu!")
                
                eps1 = st.slider("eps1 - Spasial (km)", 10, 500, 50, 5,
                               help="Radius spasial maksimum antar gempa dalam cluster")
            
            eps2 = st.slider("eps2 - Temporal (hari)", 1, 180, 30, 1,
                           help="Selang waktu maksimum antar gempa dalam cluster")
            eps3 = st.slider("eps3 - Depth (km)", 5, 200, 30, 5,
                           help="Perbedaan kedalaman maksimum") if use_depth else None
            min_samples = st.slider("min_samples", 2, 20, 5, 1,
                                  help="Jumlah minimum gempa untuk membentuk cluster")
            
            params = {
                'eps1': eps1, 'eps2': eps2, 'eps3': eps3, 
                'min_samples': min_samples, 'use_depth': use_depth, 
                'auto_eps1': auto_eps1
            }
            
        else:  # K-Means
            st.subheader("⚙️ Parameter K-Means")
            auto_k = st.checkbox("🔍 Auto K", value=False,
                               help="Otomatis mencari k optimal menggunakan Elbow Method")
            
            if auto_k:
                k_min = st.number_input("K min", 2, 20, 2)
                k_max = st.number_input("K max", 3, 20, 10)
                n_clusters = None
            else:
                n_clusters = st.slider("n_clusters", 2, 20, 3, 1)
            
            spatial_w = st.slider("Spatial Weight", 0.1, 5.0, 1.0, 0.1)
            temporal_w = st.slider("Temporal Weight", 0.1, 5.0, 1.0, 0.1)
            depth_w = st.slider("Depth Weight", 0.1, 5.0, 1.0, 0.1) if use_depth else 1.0
            
            params = {
                'n_clusters': n_clusters, 'use_depth': use_depth, 
                'spatial_weight': spatial_w, 'temporal_weight': temporal_w, 
                'depth_weight': depth_w, 'auto_k': auto_k
            }
            if auto_k:
                params['k_range'] = (k_min, k_max)
        
        st.markdown("---")
        debug_mode = st.checkbox("🔍 Debug", value=False)
        run_btn = st.button("🚀 Jalankan Clustering", type="primary", use_container_width=True)
    
    # Main Content
    if df is not None:
        st.subheader("📊 Data Preview")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Events", len(df))
        col2.metric("Mag", f"{df['magnitude'].min():.1f}-{df['magnitude'].max():.1f}")
        col3.metric("Depth", f"{df['depth'].min():.0f}-{df['depth'].max():.0f}km")
        col4.metric("Durasi", f"{(df['datetime'].max()-df['datetime'].min()).days}d")
        col5.metric("Range", f"{df['latitude'].min():.1f}°-{df['latitude'].max():.1f}°")
        
        with st.expander("🔍 Lihat Data"):
            st.dataframe(df.head(20), use_container_width=True)
        
        # Tampilkan k-distance plot jika ada
        if algorithm == "ST-DBSCAN" and 'optimal_eps1_results' in st.session_state:
            with st.expander("📊 Lihat k-Distance Plot"):
                opt_results = st.session_state['optimal_eps1_results']
                selected_k = st.selectbox(
                    "Pilih k untuk visualisasi:",
                    list(opt_results['k_results'].keys())
                )
                fig_kdist = plot_knn_distance(opt_results, selected_k=selected_k)
                st.plotly_chart(fig_kdist, use_container_width=True)
        
        st.markdown("---")
        
        # Run Clustering
        if run_btn:
            st.subheader("⚡ Proses Clustering")
            
            if algorithm == "ST-DBSCAN":
                model = STDBSCAN(
                    eps1=params['eps1'], 
                    eps2=params['eps2'], 
                    eps3=params['eps3'],
                    min_samples=params['min_samples'], 
                    use_depth=params['use_depth']
                )
                
                with st.spinner("Running ST-DBSCAN..."):
                    if params['use_depth']:
                        labels = model.fit(df['latitude'].values, df['longitude'].values,
                                         df['datetime'].values, depths=df['depth'].values, 
                                         debug=debug_mode)
                    else:
                        labels = model.fit(df['latitude'].values, df['longitude'].values,
                                         df['datetime'].values, debug=debug_mode)
                
                st.session_state.update({
                    'labels': labels, 
                    'algorithm': 'ST-DBSCAN', 
                    'params': params, 
                    'model': model, 
                    'clustering_done': True
                })
                st.success(f"✅ ST-DBSCAN selesai!")
                
            else:  # K-Means
                if params.get('auto_k'):
                    st.info("🔍 Mencari k optimal...")
                    k_range = range(int(params['k_range'][0]), int(params['k_range'][1]) + 1)
                    
                    opt_results = find_optimal_k(
                        df, k_range=k_range, 
                        use_depth=params['use_depth'],
                        spatial_weight=params['spatial_weight'],
                        temporal_weight=params['temporal_weight'],
                        depth_weight=params['depth_weight']
                    )
                    
                    st.session_state['optimal_k_results'] = opt_results
                    
                    st.markdown("### 📊 K Optimal Analysis")
                    st.plotly_chart(create_elbow_plot(opt_results), use_container_width=True)
                    
                    silhouettes = opt_results['silhouettes']
                    rec_k = opt_results['k_values'][silhouettes.index(max(silhouettes))]
                    st.success(f"💡 Rekomendasi: k = {rec_k}")
                    
                    n_clusters = st.selectbox("Pilih k:", opt_results['k_values'],
                                             index=opt_results['k_values'].index(rec_k))
                    params['n_clusters'] = n_clusters
                
                model = SpatioTemporalKMeans(
                    n_clusters=params['n_clusters'], 
                    use_depth=params['use_depth'],
                    spatial_weight=params['spatial_weight'],
                    temporal_weight=params['temporal_weight'],
                    depth_weight=params['depth_weight']
                )
                
                with st.spinner("Running K-Means..."):
                    if params['use_depth']:
                        labels = model.fit(df['latitude'].values, df['longitude'].values,
                                         df['datetime'].values, depths=df['depth'].values, 
                                         debug=debug_mode)
                    else:
                        labels = model.fit(df['latitude'].values, df['longitude'].values,
                                         df['datetime'].values, debug=debug_mode)
                
                st.session_state.update({
                    'labels': labels, 
                    'algorithm': 'K-Means',
                    'params': params, 
                    'model': model, 
                    'clustering_done': True
                })
                st.success(f"✅ K-Means selesai! k={params['n_clusters']}")
        
        # Display Results
        if st.session_state.get('clustering_done'):
            labels = st.session_state['labels']
            algo = st.session_state['algorithm']
            params = st.session_state['params']
            model = st.session_state['model']
            
            st.markdown("---")
            st.subheader("📈 Hasil Clustering")
            
            # Statistics
            if algo == "ST-DBSCAN":
                stats_df, n_clusters, n_noise, patterns = analyze_clusters(df, labels, params['eps2'])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Clusters", n_clusters)
                col2.metric("Noise", n_noise)
                col3.metric("Noise %", f"{n_noise/len(labels)*100:.1f}%")
                col4.metric("Clustered", len(labels) - n_noise)
                
                metrics = calculate_clustering_metrics(df, labels, params['eps1'], params['eps2'])
            else:
                stats_df, n_clusters, patterns = analyze_kmeans_clusters_with_patterns(df, labels)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Clusters", n_clusters)
                col2.metric("Events", len(labels))
                col3.metric("Avg Size", f"{len(labels)/n_clusters:.1f}")
                col4.metric("Inertia", f"{model.inertia_:.1f}")
                
                metrics = calculate_kmeans_metrics(df, labels, model)
            
            if metrics:
                st.markdown("### 📊 Metrik Evaluasi")
                col1, col2, col3, col4 = st.columns(4)
                if metrics.get('silhouette_score'):
                    col1.metric("Silhouette", f"{metrics['silhouette_score']:.3f}")
                if metrics.get('davies_bouldin_index'):
                    col2.metric("Davies-Bouldin", f"{metrics['davies_bouldin_index']:.3f}")
                if metrics.get('inertia'):
                    col3.metric("Inertia", f"{metrics['inertia']:.1f}")
                if metrics.get('avg_cluster_size'):
                    col4.metric("Avg Size", f"{metrics['avg_cluster_size']:.1f}")
            
            st.markdown("### 📋 Statistik Cluster")
            st.dataframe(stats_df, use_container_width=True)
            
            # Detail pola cluster
            if patterns:
                st.markdown("### 🔍 Analisis Pola Cluster")
                
                pattern_summary = {}
                for pattern in patterns:
                    p = pattern['pattern_info']['pattern']
                    pattern_summary[p] = pattern_summary.get(p, 0) + 1
                
                # Summary badges
                st.markdown("**Ringkasan Pola:**")
                cols = st.columns(len(pattern_summary))
                pattern_emojis = {
                    'OCCASIONAL': '⚡',
                    'STATIONARY': '🎯',
                    'REAPPEARING': '🔄',
                    'TRACK': '🚶',
                    'DISPERSED': '🌐',
                    'COMPACT': '🎪',
                    'UNDEFINED': '❓'
                }
                for i, (ptype, count) in enumerate(pattern_summary.items()):
                    emoji = pattern_emojis.get(ptype, '📌')
                    cols[i].metric(f"{emoji} {ptype}", count)
                
                st.markdown("---")
                
                # Detail per cluster
                for pattern in patterns:
                    cid = pattern['cluster_id']
                    pinfo = pattern['pattern_info']
                    
                    emoji = pattern_emojis.get(pinfo['pattern'], '📌')
                    
                    with st.expander(f"{emoji} Cluster {cid} - {pinfo['pattern']} (Confidence: {pinfo['confidence']}%)"):
                        st.write(f"**Deskripsi:** {pinfo['description']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Karakteristik Temporal:**")
                            st.write(f"- Durasi: {pinfo['durasi']} hari")
                            st.write(f"- Max Time Gap: {pinfo['max_time_gap']} hari")
                            st.write(f"- Mean Time Gap: {pinfo['mean_time_gap']:.1f} hari")
                            st.write(f"- Intensitas: {pinfo['temporal_intensity']:.2f} events/hari")
                        
                        with col2:
                            st.write("**Karakteristik Spasial:**")
                            st.write(f"- Max Spatial Spread: {pinfo['max_spatial_spread']:.2f} km")
                            st.write(f"- Mean Spatial Spread: {pinfo['mean_spatial_spread']:.2f} km")
                            st.write(f"- Total Movement: {pinfo['total_movement']:.2f} km")
                            st.write(f"- Jumlah Events: {pinfo['n_events']}")
                
                # Interpretasi pola
                st.markdown("---")
                st.info("""
                **Interpretasi Pola Cluster:**
                
                - ⚡ **OCCASIONAL**: Burst aktivitas dalam waktu singkat, biasanya aftershock sequence
                - 🎯 **STATIONARY**: Aktivitas kontinu di lokasi tetap, zona patahan aktif
                - 🔄 **REAPPEARING**: Aktivitas berulang dengan jeda, episodic activity
                - 🚶 **TRACK**: Pergerakan spasial, migrasi gempa atau propagasi rupture
                - 🌐 **DISPERSED**: Aktivitas tersebar luas, background seismicity
                - 🎪 **COMPACT**: Cluster padat dan aktif, foreshock/mainshock region
                - ❓ **UNDEFINED**: Pola tidak jelas atau campuran
                """)
            
            # Visualizations
            st.markdown("---")
            st.subheader("📍 Visualisasi")
            
            tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta", "⏱️ Temporal", "🎲 3D", "📊 Perbandingan"])
            
            with tab1:
                st.plotly_chart(create_spatial_plot(df, labels), use_container_width=True)
            
            with tab2:
                st.plotly_chart(create_temporal_plot(df, labels), use_container_width=True)
            
            with tab3:
                st.plotly_chart(create_3d_plot(df, labels), use_container_width=True)
            
            with tab4:
                st.plotly_chart(create_comparison_chart(df, labels), use_container_width=True)
            
            # Export
            st.markdown("---")
            st.subheader("💾 Export Hasil")
            df_export = df.copy()
            df_export['cluster'] = labels
            csv = df_export.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "earthquake_clusters.csv", "text/csv")
    
    else:
        st.info("👈 Pilih sumber data di sidebar untuk memulai")
        st.markdown("""
        ### 📖 Tentang Aplikasi
        
        **Algoritma Clustering:**
        - **ST-DBSCAN**: Density-based, deteksi noise otomatis, tidak perlu tentukan jumlah cluster
          - ✅ Auto-detect eps1 optimal dengan k-distance plot
        - **K-Means**: Partition-based, perlu tentukan k, lebih cepat untuk dataset besar
          - ✅ Auto-detect k optimal dengan Elbow Method
        
        **Mode Clustering:**
        - **3D**: Waktu + Latitude + Longitude
        - **4D**: Waktu + Latitude + Longitude + Kedalaman
        
        **Format Data CSV:** 
        ```
        datetime, latitude, longitude, magnitude, depth
        ```
        """)


if __name__ == "__main__":
    main()
